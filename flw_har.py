"""
Federated HAR training with Sensitivity-Aware Private Masking (SAPM) + DP,
simplified for calibration/sweeping.

CHANGES IN THIS VERSION (vs. the quantized/permuted version):

  1. QUANTIZATION AND PERMUTATION REMOVED.
     The previous version noised the selected values, then stochastically
     quantized them to `quant_bits` and permuted them before sending. Both
     steps are lossy distortions applied ON TOP of DP noise, and neither is
     part of the (epsilon, delta) accounting -- so they were pure signal
     loss with no privacy benefit. They're gone. The client now sends the
     sparse, clipped, noised delta as a plain dense float32 array. This
     also means the server-side reconstruct step (which used to unpermute
     + dequantize) is now just "use the array as-is" -- removed too.
     NOTE: this also means the sparse-encoding communication savings claim
     ("as-sent could be X% of baseline if sparse-encoded") no longer has a
     quantized value size to shrink -- comm_sparse_encoded_bytes now
     assumes 4-byte float values instead of quant_bits-sized ones. This
     is expected: you're trading bandwidth (dense sends the full model
     every round, same as no-privacy) for a cleaner accuracy signal while
     you calibrate DP. Re-add compression later, after DP is actually
     usable, not before.

  2. FULL CLI CONFIG for sweeping epsilon x keep_ratio without editing code.
     See `parse_args()` / the bottom of this file for the flags. Every
     run also gets a `--tag` so sweep outputs (best model, confusion
     matrix) don't clobber each other.

  3. DP_CLIP_MODE defaults to "per_tensor" with a classifier-weighted
     budget (`--classifier_weight`, default 5.0). This gives the
     classifier head a larger share of the fixed clip_norm budget than a
     plain sqrt(size)-weighted split would -- same total (epsilon, delta)
     guarantee as "global" mode, just reallocated. At startup the script
     prints exactly which tensors matched the "classifier" substring so
     you can confirm it's actually hitting your model's head before
     trusting the run. IF THE MATCH LIST IS EMPTY, FIX
     --classifier_substring BEFORE TRUSTING THIS MODE -- an empty match
     means every tensor silently falls back to weight 1.0 and you're
     running the sqrt(size)-ish default, not the weighted intent.

  4. Output array ordering fix: arrays are now placed back into
     model-state-dict key order explicitly (`{name: array}` dict, then
     re-ordered by `state_dict.keys()`) rather than relying on the
     accidental ordering of "plain tensors first, then private tensors."
     That accidental ordering happens to be correct only when the model
     has zero non-trainable buffers (e.g. no BatchNorm running stats)
     interleaved between trainable params in the state_dict. It's cheap
     to make this robust instead of assuming it, so it's fixed here.

WHAT THIS VERSION DOES NOT FIX BY ITSELF:
  The core SNR problem is dimensionality, not epsilon. In the logged runs:
    - keep_ratio=0.6, epsilon=8,  clip_norm=1.49 (lossy) -> snr ~ 0.003
    - keep_ratio=0.3, epsilon=40, clip_norm=1.49 (slack)  -> snr ~ 0.007-0.009
  A 5x epsilon increase (which cuts sigma 5x) barely moved SNR, because
  halving keep_ratio also roughly halved k_total (920K -> 460K), and
  per-coordinate signal scales like raw_norm/sqrt(k) -- so the dominant
  lever is k (via keep_ratio), not epsilon alone. Use the sweep below to
  find out how much lower keep_ratio needs to go; don't assume epsilon
  alone will get you to a usable SNR.

HOW TO SWEEP (bash, from repo root):
    for eps in 8 20 40; do
      for kr in 0.6 0.3 0.1 0.05; do
        python sapm_dp_train.py \
          --epsilon $eps --keep_ratio $kr \
          --clip_norm 1.2 \
          --num_rounds 5 \
          --tag "eps${eps}_kr${kr}"
      done
    done
  Run short (--num_rounds 5) first to read the `[dp-calib]` and `snr`
  lines cheaply, THEN commit to a full --num_rounds 40 run for the
  config that looks best. Re-check clip_norm calibration whenever you
  change keep_ratio -- raw_norm depends on it.
"""

import argparse
import json
import time
import warnings

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset


# ====================== PRIVACY (SAPM) HELPERS ======================
class pu:
    @staticmethod
    def compute_topk_mask(fisher_flat: np.ndarray, keep_ratio: float) -> np.ndarray:
        n = fisher_flat.size
        k = max(1, int(np.ceil(keep_ratio * n)))
        if k >= n:
            return np.ones(n, dtype=bool)
        idx = np.argpartition(fisher_flat, -k)[-k:]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask

    # ---------------- DP helpers ----------------
    @staticmethod
    def clip_l2(v: np.ndarray, clip_norm: float):
        """Clip a flat vector to L2 norm <= clip_norm. Returns (clipped_v, scale_used)."""
        norm = float(np.linalg.norm(v))
        if norm <= clip_norm or norm == 0.0:
            return v, 1.0
        scale = clip_norm / norm
        return v * scale, scale

    @staticmethod
    def gaussian_mechanism_sigma(clip_norm: float, epsilon: float, delta: float) -> float:
        """
        Classic (epsilon, delta)-DP calibration for the Gaussian mechanism
        with L2 sensitivity `clip_norm`:

            sigma = clip_norm * sqrt(2 * ln(1.25 / delta)) / epsilon

        Standard textbook calibration (Dwork & Roth). Sufficient, not
        necessarily tight -- a tighter analytic Gaussian mechanism
        (Balle & Wang, 2018) could be substituted without changing
        anything else in this pipeline.
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("epsilon must be > 0 and delta must be in (0, 1).")
        return float(clip_norm * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)

    @staticmethod
    def add_gaussian_noise(v: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return v
        return v + np.random.normal(loc=0.0, scale=sigma, size=v.shape).astype(v.dtype)

    @staticmethod
    def per_tensor_clip_norms(names, size_by_name, total_clip_norm, weights=None):
        """
        Split one total L2 clip budget `total_clip_norm` into a per-tensor
        budget C_l for each tensor in `names`, such that

            sqrt(sum_l C_l^2) == total_clip_norm

        This keeps the PRIVACY GUARANTEE IDENTICAL to clipping the whole
        concatenated vector at `total_clip_norm`: if each tensor's
        selected-value vector v_l is independently clipped to
        ||v_l|| <= C_l, then the full vector v = (v_1, ..., v_L) satisfies
        ||v||^2 = sum ||v_l||^2 <= sum C_l^2 = total_clip_norm^2, i.e. the
        same L2 sensitivity bound as before. Adding the SAME sigma
        (calibrated from total_clip_norm, epsilon, delta) independently to
        each tensor is then exactly the multivariate Gaussian mechanism
        over the full vector. Nothing about (epsilon, delta) changes --
        only which coordinates get more/less of the fixed clip budget.

        weights: optional dict {tensor_name_substring: weight}. A tensor's
        weight is `weights[s]` for the FIRST substring `s` in `weights`
        that appears in the tensor's name; tensors matching no substring
        default to weight 1.0. If `weights` is None, weight defaults to
        sqrt(size) for every tensor (close to what "global" mode does
        under a roughly-uniform-magnitude assumption).
        """
        if weights is None:
            shares = np.array([np.sqrt(max(1, size_by_name[n])) for n in names], dtype=np.float64)
        else:
            def weight_for(name):
                for substr, w in weights.items():
                    if substr in name:
                        return float(w)
                return 1.0
            shares = np.array([weight_for(n) for n in names], dtype=np.float64)
        norm = np.sqrt(np.sum(shares ** 2))
        if norm == 0:
            shares = np.ones(len(names), dtype=np.float64)
            norm = np.sqrt(len(names))
        shares = shares / norm  # now sqrt(sum(shares^2)) == 1
        return {n: float(total_clip_norm * s) for n, s in zip(names, shares)}


# ====================== CLI CONFIG ======================
def parse_args():
    p = argparse.ArgumentParser(description="SAPM+DP federated HAR training (calibration build)")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--test_csv", type=str, default="test.csv")
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--num_rounds", type=int, default=40)

    p.add_argument("--use_privacy", action="store_true", default=True)
    p.add_argument("--no_privacy", dest="use_privacy", action="store_false")
    p.add_argument("--keep_ratio", type=float, default=0.3,
                    help="Fraction of each tensor's elements selected by Fisher top-k.")

    p.add_argument("--use_dp_noise", action="store_true", default=True)
    p.add_argument("--no_dp_noise", dest="use_dp_noise", action="store_false")
    p.add_argument("--epsilon", type=float, default=40.0, help="Per-round epsilon.")
    p.add_argument("--delta", type=float, default=1e-5, help="Per-round delta.")
    p.add_argument("--clip_norm", type=float, default=1.2,
                    help="L2 clip budget over the selected-value vector. RECALIBRATE this "
                         "whenever you change keep_ratio -- watch the [dp-calib] line and "
                         "set this to the reported avg pre-clip L2 norm.")
    p.add_argument("--clip_mode", type=str, default="per_tensor", choices=["global", "per_tensor"])
    p.add_argument("--classifier_weight", type=float, default=5.0,
                    help="Clip-budget weight multiplier for tensors matching "
                         "--classifier_substring, used only when --clip_mode per_tensor.")
    p.add_argument("--classifier_substring", type=str, default="imu_head",
                    help="Substring matched against param names to find the classifier head. "
                         "Defaults to 'imu_head' -- this model's actual head layer name, "
                         "confirmed from the printed parameter list ('classifier' matched "
                         "nothing). Pass 'imu_head.4' instead to weight only the final "
                         "linear layer of the head rather than the whole head block. "
                         "VERIFY this actually matches something at startup -- the script "
                         "prints the matched tensor names before training begins.")

    p.add_argument("--index_bytes_per_element", type=int, default=4,
                    help="Bytes per sparse-encoded index (int32), used only for the "
                         "comm_sparse_encoded_bytes estimate, not for what's actually sent.")

    p.add_argument("--tag", type=str, default="run",
                    help="Suffix for output files (best_model_<tag>.pth, "
                         "final_confusion_matrix_<tag>.png) so sweep runs don't overwrite "
                         "each other.")

    args, _unknown = p.parse_known_args()  # parse_known_args: tolerate notebook/Colab kernel args
    return args


ARGS = parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

NUM_CLIENTS = ARGS.num_clients
LOCAL_EPOCHS = ARGS.local_epochs
NUM_ROUNDS = ARGS.num_rounds

USE_PRIVACY = ARGS.use_privacy
PRIVACY_KEEP_RATIO = ARGS.keep_ratio

USE_DP_NOISE = ARGS.use_dp_noise
PRIVACY_DP_EPSILON = ARGS.epsilon
PRIVACY_DP_DELTA = ARGS.delta
PRIVACY_DP_CLIP_NORM = ARGS.clip_norm

DP_CLIP_MODE = ARGS.clip_mode
DP_TENSOR_WEIGHTS = {ARGS.classifier_substring: ARGS.classifier_weight} if ARGS.clip_mode == "per_tensor" else None

INDEX_BYTES_PER_ELEMENT = ARGS.index_bytes_per_element
TAG = ARGS.tag

print(f"Using device: {DEVICE}")
print(f"Privacy strategy: SAPM (no quant/permute) | enabled={USE_PRIVACY} | keep_ratio={PRIVACY_KEEP_RATIO}")
print(f"DP add-on: enabled={USE_DP_NOISE} | epsilon(per-round)={PRIVACY_DP_EPSILON} | "
      f"delta(per-round)={PRIVACY_DP_DELTA} | clip_norm={PRIVACY_DP_CLIP_NORM} | "
      f"clip_mode={DP_CLIP_MODE}"
      + (f" | classifier_weight={ARGS.classifier_weight} (substring='{ARGS.classifier_substring}')"
         if DP_CLIP_MODE == "per_tensor" else ""))
print(f"Run tag: {TAG}")


# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    test_dataset = IMUDataset(test_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset


def split_train_data(train_dataset, num_clients=NUM_CLIENTS, seed=42):
    n = len(train_dataset)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    client_datasets = []
    size = n // num_clients
    print(f"\n=== Client Data Distribution (Seed={seed}) ===")
    for i in range(num_clients):
        start = i * size
        end = start + size if i < num_clients - 1 else n
        subset = Subset(train_dataset, indices[start:end])
        client_datasets.append(subset)
        print(f"Client {i} -> {len(subset)} samples")
    print("=" * 60)
    return client_datasets


# ====================== CLIENT ======================
class IMUClient(fl.client.NumPyClient):
    def __init__(self, train_subset):
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if hasattr(parameters, "tensors"):
            params = parameters_to_ndarrays(parameters)
        else:
            params = parameters
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, fit_config):
        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        use_privacy = fit_config.get("use_privacy", USE_PRIVACY)
        keep_ratio = fit_config.get("privacy_keep_ratio", PRIVACY_KEEP_RATIO)

        use_dp_noise = fit_config.get("use_dp_noise", USE_DP_NOISE) and use_privacy
        dp_epsilon = fit_config.get("privacy_dp_epsilon", PRIVACY_DP_EPSILON)
        dp_delta = fit_config.get("privacy_dp_delta", PRIVACY_DP_DELTA)
        dp_clip_norm = fit_config.get("privacy_dp_clip_norm", PRIVACY_DP_CLIP_NORM)
        dp_clip_mode = fit_config.get("dp_clip_mode", DP_CLIP_MODE)
        # Flower's FitIns.config only accepts scalar types (bool/bytes/float/int/str) --
        # a raw dict here throws inside the Ray worker with zero useful traceback, which
        # is what "aggregate_fit: received 0 results and N failures" every round means.
        # The server JSON-encodes this dict before putting it in config; decode it back here.
        _dp_tensor_weights_raw = fit_config.get("dp_tensor_weights", None)
        if _dp_tensor_weights_raw is None:
            dp_tensor_weights = DP_TENSOR_WEIGHTS
        elif isinstance(_dp_tensor_weights_raw, str):
            dp_tensor_weights = json.loads(_dp_tensor_weights_raw)
        else:
            dp_tensor_weights = _dp_tensor_weights_raw

        self.model.train()
        total_loss = 0.0

        fisher_accum = {
            name: torch.zeros_like(p)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        n_grad_steps = 0
        fisher_time_sec = 0.0

        for _ in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()

                self.optimizer.zero_grad()
                output = self.model({"imu": imu})
                loss = self.criterion(output, label)
                loss.backward()

                _fisher_start = time.perf_counter()
                for name, p in self.model.named_parameters():
                    if p.grad is not None and name in fisher_accum:
                        fisher_accum[name] += p.grad.detach() ** 2
                fisher_time_sec += time.perf_counter() - _fisher_start
                n_grad_steps += 1

                self.optimizer.step()
                total_loss += loss.item()

        for k in fisher_accum:
            fisher_accum[k] /= max(1, n_grad_steps)

        new_state = self.model.state_dict()
        state_keys_in_order = list(new_state.keys())  # authoritative order for output

        comm_no_privacy_bytes = 0

        plain_names = []
        plain_arrays = {}
        priv_names = []
        priv_delta_flat = {}
        priv_mask = {}
        priv_shape = {}

        for name, new_val in new_state.items():
            old_val = old_state[name]
            delta = (new_val - old_val).cpu().numpy()
            comm_no_privacy_bytes += delta.astype(np.float32).nbytes

            if not use_privacy or name not in fisher_accum:
                plain_names.append(name)
                plain_arrays[name] = delta.astype(np.float32)
                continue

            fisher_flat = fisher_accum[name].cpu().numpy().reshape(-1)
            delta_flat = delta.reshape(-1).astype(np.float32)
            mask = pu.compute_topk_mask(fisher_flat, keep_ratio)

            priv_names.append(name)
            priv_delta_flat[name] = delta_flat
            priv_mask[name] = mask
            priv_shape[name] = delta.shape

        # ===== PASS 2: clip + Gaussian noise (global or per-tensor budget) =====
        dp_time_sec = 0.0
        dp_scale = 1.0
        dp_scale_min = 1.0
        dp_scale_max = 1.0
        dp_sigma = 0.0
        dp_k_total = 0
        dp_signal_mean_abs = 0.0
        dp_raw_norm = 0.0  # pre-clip L2 norm of the whole selected-value vector,
                            # reported the same way regardless of clip_mode so the
                            # two modes stay directly comparable in [dp-calib].

        noised_selected = {}

        if priv_names:
            _dp_t0 = time.perf_counter()

            per_name_selected = {name: priv_delta_flat[name][priv_mask[name]] for name in priv_names}
            lengths = [per_name_selected[name].size for name in priv_names]
            size_by_name = dict(zip(priv_names, lengths))
            dp_k_total = int(sum(lengths))

            if use_dp_noise and dp_k_total > 0:
                global_vec = np.concatenate([per_name_selected[name] for name in priv_names]).astype(np.float32)
                dp_signal_mean_abs = float(np.mean(np.abs(global_vec))) if global_vec.size else 0.0
                dp_raw_norm = float(np.linalg.norm(global_vec))

                # sigma is ALWAYS calibrated from the TOTAL clip_norm, regardless of
                # clip_mode -- this is what keeps (epsilon, delta) identical between
                # "global" and "per_tensor".
                dp_sigma = pu.gaussian_mechanism_sigma(dp_clip_norm, dp_epsilon, dp_delta)

                if dp_clip_mode == "per_tensor":
                    per_tensor_clip = pu.per_tensor_clip_norms(
                        priv_names, size_by_name, dp_clip_norm, weights=dp_tensor_weights
                    )
                    scales = []
                    for name in priv_names:
                        v = per_name_selected[name]
                        c_l = per_tensor_clip[name]
                        clipped_v, scale_l = pu.clip_l2(v, c_l)
                        noised_selected[name] = pu.add_gaussian_noise(clipped_v, dp_sigma).astype(np.float32)
                        scales.append(scale_l)
                    dp_scale = float(np.mean(scales)) if scales else 1.0
                    dp_scale_min = float(np.min(scales)) if scales else 1.0
                    dp_scale_max = float(np.max(scales)) if scales else 1.0
                else:  # "global" -- one joint clip over the concatenated vector
                    clipped_vec, dp_scale = pu.clip_l2(global_vec, dp_clip_norm)
                    dp_scale_min = dp_scale_max = dp_scale
                    noised_vec = pu.add_gaussian_noise(clipped_vec, dp_sigma).astype(np.float32)
                    offset = 0
                    for name, length in zip(priv_names, lengths):
                        noised_selected[name] = noised_vec[offset:offset + length]
                        offset += length
            else:
                for name in priv_names:
                    noised_selected[name] = per_name_selected[name]

            dp_time_sec = time.perf_counter() - _dp_t0

        # ===== PASS 3: scatter noised values back into dense arrays (no quant/permute) =====
        out_by_name = {}
        nz_total, elem_total = 0, 0
        comm_dense_bytes = 0
        comm_sparse_encoded_bytes = 0

        for name in plain_names:
            arr = plain_arrays[name]
            out_by_name[name] = arr
            nz_total += np.count_nonzero(arr)
            elem_total += arr.size
            comm_dense_bytes += arr.nbytes
            comm_sparse_encoded_bytes += arr.nbytes

        for name in priv_names:
            mask = priv_mask[name]
            shape = priv_shape[name]

            sparse_delta = np.zeros_like(priv_delta_flat[name])
            sparse_delta[mask] = noised_selected[name]
            sparse_delta = sparse_delta.reshape(shape).astype(np.float32)
            out_by_name[name] = sparse_delta

            nz_count = int(np.count_nonzero(mask))
            nz_total += np.count_nonzero(sparse_delta)
            elem_total += sparse_delta.size

            comm_dense_bytes += sparse_delta.nbytes
            # no quantization now -> 4 bytes/value for the sparse-encoded estimate
            comm_sparse_encoded_bytes += nz_count * (INDEX_BYTES_PER_ELEMENT + 4)

        # Explicit reorder into state_dict key order -- don't rely on
        # "plain tensors happened to come first" being true for this model.
        out_arrays = [out_by_name[name] for name in state_keys_in_order]

        dp_snr = (dp_signal_mean_abs / dp_sigma) if dp_sigma > 0 else float("inf")

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "nonzero_ratio": float(nz_total / max(1, elem_total)),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_sparse_encoded_bytes": comm_sparse_encoded_bytes,
            "comm_no_privacy_bytes": comm_no_privacy_bytes,
            "fisher_time_sec": fisher_time_sec,
            "dp_time_sec": dp_time_sec,
            "dp_clip_norm": dp_clip_norm,
            "dp_epsilon": dp_epsilon,
            "dp_delta": dp_delta,
            "dp_scale": dp_scale,
            "dp_scale_min": dp_scale_min,
            "dp_scale_max": dp_scale_max,
            "dp_sigma": dp_sigma,
            "dp_k_total": dp_k_total,
            "dp_signal_mean_abs": dp_signal_mean_abs,
            "dp_snr": dp_snr,
            "dp_raw_norm": dp_raw_norm,
            "dp_clip_mode": dp_clip_mode,
        }
        return out_arrays, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, eval_config):
        self.set_parameters(parameters)
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()
                output = self.model({"imu": imu})
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return float(0.0), len(self.train_loader.dataset), {"accuracy": accuracy}


# ====================== STRATEGY ======================
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, use_privacy=USE_PRIVACY,
                 privacy_keep_ratio=PRIVACY_KEEP_RATIO,
                 use_dp_noise=USE_DP_NOISE,
                 privacy_dp_epsilon=PRIVACY_DP_EPSILON,
                 privacy_dp_delta=PRIVACY_DP_DELTA,
                 privacy_dp_clip_norm=PRIVACY_DP_CLIP_NORM,
                 dp_clip_mode=DP_CLIP_MODE,
                 dp_tensor_weights=DP_TENSOR_WEIGHTS,
                 **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_privacy = use_privacy
        self.privacy_keep_ratio = privacy_keep_ratio
        self.use_dp_noise = use_dp_noise
        self.privacy_dp_epsilon = privacy_dp_epsilon
        self.privacy_dp_delta = privacy_dp_delta
        self.privacy_dp_clip_norm = privacy_dp_clip_norm
        self.dp_clip_mode = dp_clip_mode
        self.dp_tensor_weights = dp_tensor_weights

        self.total_comm_dense_bytes = 0
        self.total_comm_sparse_encoded_bytes = 0
        self.total_comm_no_privacy_bytes = 0
        self.total_fisher_time_sec = 0.0
        self.total_dp_time_sec = 0.0
        self.dp_sigma_history = []
        self.dp_snr_history = []
        self.dp_k_total_history = []
        self.dp_raw_norm_history = []

        if self.dp_clip_mode == "per_tensor" and self.dp_tensor_weights:
            matched = [n for n, _ in self.global_model.named_parameters()
                       if any(s in n for s in self.dp_tensor_weights)]
            print(f"\n[per_tensor clip_mode] classifier weight={self.dp_tensor_weights} "
                  f"matched {len(matched)} tensor(s):")
            for n in matched:
                print(f"    {n}")
            if not matched:
                print("    *** WARNING: NO TENSORS MATCHED. The classifier substring "
                      "doesn't hit any param name -- every tensor is falling back to "
                      "weight 1.0. Check --classifier_substring against your model's "
                      "actual layer names (see the list below). ***")
                print("  Full parameter name list:")
                for n, _ in self.global_model.named_parameters():
                    print(f"    {n}")

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_privacy"] = self.use_privacy
            fit_ins.config["privacy_keep_ratio"] = self.privacy_keep_ratio
            fit_ins.config["privacy_seed"] = server_round
            fit_ins.config["use_dp_noise"] = self.use_dp_noise
            fit_ins.config["privacy_dp_epsilon"] = self.privacy_dp_epsilon
            fit_ins.config["privacy_dp_delta"] = self.privacy_dp_delta
            fit_ins.config["privacy_dp_clip_norm"] = self.privacy_dp_clip_norm
            fit_ins.config["dp_clip_mode"] = self.dp_clip_mode
            if self.dp_tensor_weights is not None:
                # JSON-encode: Flower's FitIns.config values must be scalar
                # (bool/bytes/float/int/str), not dict -- see client-side note in fit().
                fit_ins.config["dp_tensor_weights"] = json.dumps(self.dp_tensor_weights)
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0
        nz_ratios = []

        round_comm_dense_bytes = 0
        round_comm_sparse_encoded_bytes = 0
        round_comm_no_privacy_bytes = 0
        round_fisher_time_sec = []

        round_dp_time_sec = []
        round_dp_sigma = []
        round_dp_snr = []
        round_dp_k_total = []
        round_dp_scale = []
        round_dp_scale_min = []
        round_dp_scale_max = []
        round_dp_raw_norm = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            nz_ratios.append(fit_res.metrics.get("nonzero_ratio", 1.0))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_sparse_encoded_bytes += fit_res.metrics.get("comm_sparse_encoded_bytes", 0)
            round_comm_no_privacy_bytes += fit_res.metrics.get("comm_no_privacy_bytes", 0)
            round_fisher_time_sec.append(fit_res.metrics.get("fisher_time_sec", 0.0))

            round_dp_time_sec.append(fit_res.metrics.get("dp_time_sec", 0.0))
            round_dp_sigma.append(fit_res.metrics.get("dp_sigma", 0.0))
            round_dp_snr.append(fit_res.metrics.get("dp_snr", 0.0))
            round_dp_k_total.append(fit_res.metrics.get("dp_k_total", 0))
            round_dp_scale.append(fit_res.metrics.get("dp_scale", 1.0))
            round_dp_scale_min.append(fit_res.metrics.get("dp_scale_min", 1.0))
            round_dp_scale_max.append(fit_res.metrics.get("dp_scale_max", 1.0))
            round_dp_raw_norm.append(fit_res.metrics.get("dp_raw_norm", 0.0))

            # No unpermute/dequantize anymore -- arrays are used directly.
            # `arrays` already arrives in state_dict key order (client sends
            # them explicitly reordered by state_keys_in_order).
            for k, arr in zip(keys, arrays):
                weighted_deltas[k] += arr.reshape(global_state[k].shape).astype(np.float64) * num_examples

            total_examples += num_examples

        new_state = {}
        for k in keys:
            avg_delta = weighted_deltas[k] / max(1, total_examples)
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype, device=global_state[k].device)

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_nz = float(np.mean(nz_ratios)) if nz_ratios else 1.0

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_sparse_encoded_bytes += round_comm_sparse_encoded_bytes
        self.total_comm_no_privacy_bytes += round_comm_no_privacy_bytes
        avg_fisher_time = float(np.mean(round_fisher_time_sec)) if round_fisher_time_sec else 0.0
        self.total_fisher_time_sec += avg_fisher_time

        avg_dp_time = float(np.mean(round_dp_time_sec)) if round_dp_time_sec else 0.0
        avg_dp_sigma = float(np.mean(round_dp_sigma)) if round_dp_sigma else 0.0
        finite_snrs = [s for s in round_dp_snr if np.isfinite(s)]
        avg_dp_snr = float(np.mean(finite_snrs)) if finite_snrs else float("inf")
        avg_dp_k_total = float(np.mean(round_dp_k_total)) if round_dp_k_total else 0.0
        avg_dp_scale = float(np.mean(round_dp_scale)) if round_dp_scale else 1.0
        avg_dp_raw_norm = float(np.mean(round_dp_raw_norm)) if round_dp_raw_norm else 0.0
        avg_dp_scale_min = float(np.mean(round_dp_scale_min)) if round_dp_scale_min else 1.0
        avg_dp_scale_max = float(np.mean(round_dp_scale_max)) if round_dp_scale_max else 1.0
        self.total_dp_time_sec += avg_dp_time
        self.dp_sigma_history.append(avg_dp_sigma)
        self.dp_snr_history.append(avg_dp_snr)
        self.dp_k_total_history.append(avg_dp_k_total)
        self.dp_raw_norm_history.append(avg_dp_raw_norm)

        compression_vs_no_privacy = (
            round_comm_sparse_encoded_bytes / round_comm_no_privacy_bytes
            if round_comm_no_privacy_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | Avg transmitted nonzero ratio: {avg_nz:.3f}")
        print(f"  [comm] as-sent (dense): {round_comm_dense_bytes/1e6:.3f} MB | "
              f"sparse-encoded (achievable, no quant): {round_comm_sparse_encoded_bytes/1e6:.3f} MB "
              f"({compression_vs_no_privacy*100:.1f}% of no-privacy baseline) | "
              f"no-privacy baseline: {round_comm_no_privacy_bytes/1e6:.3f} MB")
        print(f"  [compute] avg client fisher_time: {avg_fisher_time*1000:.2f}ms | "
              f"avg client dp_time: {avg_dp_time*1000:.2f}ms")
        scale_spread = (f" (range {avg_dp_scale_min:.4f}-{avg_dp_scale_max:.4f} across tensors)"
                        if self.dp_clip_mode == "per_tensor" else "")
        print(f"  [dp] mode={self.dp_clip_mode} | k_total={avg_dp_k_total:.0f} | "
              f"clip_scale={avg_dp_scale:.4f}{scale_spread} | "
              f"sigma={avg_dp_sigma:.6g} | snr={avg_dp_snr:.3f} "
              f"(epsilon={self.privacy_dp_epsilon}, delta={self.privacy_dp_delta}, "
              f"clip_norm={self.privacy_dp_clip_norm})")
        print(f"  [dp-calib] avg pre-clip L2 norm of selected values: {avg_dp_raw_norm:.4f} "
              f"(current clip_norm={self.privacy_dp_clip_norm} -> "
              f"{'clipping is lossy, raise clip_norm' if avg_dp_raw_norm > self.privacy_dp_clip_norm * 1.05 else 'clip_norm has slack, could lower it to cut sigma'})")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), f"best_model_{TAG}.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.print_overhead_summary()

        return aggregated_params, {
            "accuracy": acc,
            "avg_nonzero_ratio": avg_nz,
            "comm_dense_bytes": round_comm_dense_bytes,
            "comm_sparse_encoded_bytes": round_comm_sparse_encoded_bytes,
            "dp_sigma": avg_dp_sigma,
            "dp_snr": avg_dp_snr,
            "dp_raw_norm": avg_dp_raw_norm,
        }

    def print_overhead_summary(self):
        print("\n========== OVERHEAD SUMMARY (SAPM+DP, cumulative over the run) ==========")
        print(f"Total communication AS ACTUALLY SENT (dense arrays)      : "
              f"{self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication IF SPARSE-ENCODED (achievable, no quant): "
              f"{self.total_comm_sparse_encoded_bytes/1e6:.2f} MB")
        print(f"Total communication with NO privacy (dense, keep_ratio=1) : "
              f"{self.total_comm_no_privacy_bytes/1e6:.2f} MB")
        if self.total_comm_no_privacy_bytes:
            print(f"  -> as-sent is {self.total_comm_dense_bytes/self.total_comm_no_privacy_bytes*100:.1f}% "
                  f"of no-privacy baseline (no bandwidth savings until sparse-encoding is "
                  f"actually implemented on the wire)")
            print(f"  -> sparse-encoded WOULD BE {self.total_comm_sparse_encoded_bytes/self.total_comm_no_privacy_bytes*100:.1f}% "
                  f"of no-privacy baseline if implemented")
        print(f"Total client-side fisher accumulation time (avg client, summed over rounds): "
              f"{self.total_fisher_time_sec:.2f}s")
        print(f"Total client-side DP clip+noise time (avg client, summed over rounds)      : "
              f"{self.total_dp_time_sec:.2f}s")

        print("\n---- DP (SAPM+DP hybrid) summary ----")
        print(f"Per-round budget used: epsilon={self.privacy_dp_epsilon}, delta={self.privacy_dp_delta}, "
              f"clip_norm={self.privacy_dp_clip_norm}, clip_mode={self.dp_clip_mode}")
        naive_total_epsilon = self.privacy_dp_epsilon * NUM_ROUNDS
        print(f"Naive BASIC-COMPOSITION total epsilon over {NUM_ROUNDS} rounds: {naive_total_epsilon:.2f} "
              f"(crude, conservative upper bound -- use an RDP/moments accountant for a tight bound)")
        if self.dp_sigma_history:
            print(f"Noise std sigma used per round: {self.dp_sigma_history[0]:.6g} "
                  f"(constant across rounds -- depends only on clip_norm/epsilon/delta, NOT on k or D)")
        if self.dp_k_total_history:
            avg_k = float(np.mean(self.dp_k_total_history))
            print(f"Average number of coordinates actually noised per round (k_total): {avg_k:.0f}")
        if self.dp_raw_norm_history:
            arr = np.array(self.dp_raw_norm_history)
            print(f"Pre-clip L2 norm of selected values across rounds: "
                  f"min={arr.min():.4f} avg={arr.mean():.4f} max={arr.max():.4f}")
            print(f"  -> SUGGESTED clip_norm to try next: ~{arr.mean():.4f} "
                  f"(currently set to {self.privacy_dp_clip_norm}).")
        finite_snrs = [s for s in self.dp_snr_history if np.isfinite(s)]
        if finite_snrs:
            avg_snr = float(np.mean(finite_snrs))
            print(f"Average empirical signal-to-noise ratio (mean|selected value| / sigma): {avg_snr:.4f}")
            if avg_snr < 0.5:
                needed_k_shrink = (avg_snr and (0.5 / avg_snr) ** 2) or None
                print(f"  -> SNR well below a usable range (target roughly >= 0.5-1.0). Since "
                      f"per-coordinate signal scales ~ raw_norm/sqrt(k), reaching SNR~0.5 from "
                      f"here would need k to shrink by roughly {needed_k_shrink:.0f}x if nothing "
                      f"else changes -- i.e. keep_ratio is very likely the lever to pull next, "
                      f"not epsilon. Try keep_ratio in the 0.01-0.1 range and re-check this line.")

    def evaluate_global(self, final=False):
        self.global_model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()
                outputs = self.global_model({"imu": imu})
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)

        if not final:
            return accuracy

        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("\nClassification Report")
        print(classification_report(all_labels, all_preds, zero_division=0))

        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix")
        print(cm)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Final Confusion Matrix (SAPM+DP, {TAG})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"final_confusion_matrix_{TAG}.png")
        plt.close()

        return accuracy


# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset, NUM_CLIENTS, seed=42)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    def client_fn(context):
        if hasattr(context, "node_id"):
            cid = int(context.node_id)
        elif hasattr(context, "node_config") and "cid" in context.node_config:
            cid = int(context.node_config["cid"])
        else:
            cid = 0
        client_idx = cid % len(client_datasets)
        return IMUClient(client_datasets[client_idx]).to_client()

    strategy = SaveModelStrategy(
        test_loader=test_loader,
        use_privacy=USE_PRIVACY,
        privacy_keep_ratio=PRIVACY_KEEP_RATIO,
        use_dp_noise=USE_DP_NOISE,
        privacy_dp_epsilon=PRIVACY_DP_EPSILON,
        privacy_dp_delta=PRIVACY_DP_DELTA,
        privacy_dp_clip_norm=PRIVACY_DP_CLIP_NORM,
        dp_clip_mode=DP_CLIP_MODE,
        dp_tensor_weights=DP_TENSOR_WEIGHTS,
    )

    print(f"Starting FL | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
    )


if __name__ == "__main__":
    main(ARGS.train_csv, ARGS.test_csv)