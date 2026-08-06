"""
Federated HAR training with Sensitivity-Aware Private Masking (SAPM),
now extended with a SAPM+DP hybrid.

Drop-in replacement for the original fl_train.py. Model, dataset, and
overall FL loop are unchanged; the client no longer sends raw updated
weights -- it sends a Fisher-sparsified, DP-noised, quantized,
seed-permuted delta. The server strategy reverses the quantization and
permutation before aggregating (the DP noise is NOT reversed -- that's
the whole point of adding it: it has to survive to the aggregate).

    USE_PRIVACY           -- master on/off switch for SAPM (False =
                              behaves like the original plain FedAvg
                              script; DP is also skipped in that case,
                              since there is no mask to hang it on)
    PRIVACY_KEEP_RATIO     -- fraction of each tensor's elements sent
                              per round (Fisher top-k), e.g. 0.3 = 30%
    PRIVACY_QUANT_BITS     -- bits used for stochastic quantization of
                              the transmitted values (8 is a reasonable
                              default; try 4 for a more aggressive test)

    USE_DP_NOISE           -- master on/off switch for the new DP
                              add-on. When False, this script behaves
                              exactly like the original SAPM script
                              (top-k mask + quantize + permute, no
                              noise).
    PRIVACY_DP_EPSILON,
    PRIVACY_DP_DELTA       -- per-round (epsilon, delta) budget for the
                              Gaussian mechanism. NOTE: this is a
                              PER-ROUND budget, not a total-training
                              guarantee -- composing NUM_ROUNDS
                              applications of the mechanism needs a
                              proper accountant (RDP/moments accountant)
                              for a tight bound. This script reports
                              a naive basic-composition number
                              (num_rounds * epsilon) purely as a rough,
                              conservative diagnostic -- see
                              print_overhead_summary().
    PRIVACY_DP_CLIP_NORM   -- L2 clip norm C applied to the vector of
                              *selected* (top-k) values of a client's
                              entire update, before noise is added.
                              This is the DP "sensitivity" bound: it
                              caps how much any single client's update
                              can move the aggregate, which is what the
                              noise is calibrated against.

WHY NOISE ONLY THE TOP-K VALUES (THE HYBRID IDEA):
    A naive "just bolt DP onto FedAvg" design clips and noises the
    FULL D-dimensional parameter delta (D ~ 1.5M here). For a fixed
    L2 clip norm C, spreading that clip budget over ~1.5M coordinates
    means the *signal* per coordinate is typically tiny, so a noise
    std sigma calibrated from (C, epsilon, delta) can swamp it -- this
    is why "full DP" runs often need epsilon in the thousands before
    the signal survives.

    SAPM already throws away the (1 - keep_ratio) least Fisher-
    sensitive coordinates every round. If DP noise is added only to
    the k = keep_ratio * D coordinates SAPM actually keeps (which are,
    by construction, the highest-Fisher-sensitivity, typically
    largest-magnitude ones), the *same total clip budget C* is
    concentrated over far fewer coordinates. Two things follow:
      1. The per-coordinate noise std sigma, calibrated from
         (C, epsilon, delta) via the Gaussian mechanism, does NOT
         depend on k or D directly -- it only depends on C, epsilon,
         delta. So sigma itself is unchanged by sparsification.
      2. What DOES change is the per-coordinate SIGNAL: clipping the
         same C over fewer, larger-magnitude coordinates leaves each
         surviving coordinate with a bigger share of the energy budget
         than clipping it over the full dense vector would. That raises
         the empirical signal-to-noise ratio (mean(|value|) / sigma)
         for the same (C, epsilon, delta) -- i.e. you can plausibly get
         away with a smaller epsilon (more privacy) at the same
         perceived signal quality, precisely because SAPM already did
         the work of concentrating the update onto fewer, more
         important coordinates.
    This script instruments exactly that comparison: dp_snr per round,
    and a summary that contrasts sparsified vs. dense (keep_ratio=1.0)
    settings when you re-run with different PRIVACY_KEEP_RATIO values.

OVERHEAD INSTRUMENTATION:
    Two kinds of cost are tracked every round, on both sides:

    COMMUNICATION -- three different byte counts are reported per round,
    because they answer three different questions:
      - comm_dense_bytes: what ACTUALLY crosses the wire in THIS
        implementation. The client builds a full D-length array per
        tensor with (1-keep_ratio) of it zeroed out, and Flower
        serializes that whole dense array -- so, AS CURRENTLY CODED,
        SAPM's keep_ratio does NOT reduce bytes transmitted at all
        versus sending everything. This is worth seeing plainly rather
        than assuming sparsification implies bandwidth savings.
      - comm_sparse_encoded_bytes: what a tensor WOULD cost if the
        nonzero entries were actually sparse-encoded (index + quantized
        value pairs) instead of sent as a dense array -- i.e. what
        SAPM's design is presumably meant to achieve, and the fair
        number to compare against secagg_topk.py's K-length vectors
        (which really are only K values, not a zero-padded D-length
        array). This is NOT what this script transmits today; it is
        reported as the achievable target if sparse-encoding were added.
      - comm_no_privacy_bytes: a plain dense float32 send of the same
        tensor (keep_ratio=1.0 equivalent), as a reference baseline for
        computing a compression ratio.

    COMPUTATION -- wall-clock time added by the privacy mechanism itself,
    kept separate from ordinary training time:
      - fisher_time_sec (client): time spent accumulating the per-
        parameter Fisher (squared-gradient) sensitivity signal across
        all local steps -- overhead that plain FedAvg training does not
        pay at all.
      - dp_time_sec (client, NEW): time spent clipping the global
        selected-value vector and drawing/adding Gaussian noise to it.
        Kept separate from transform_time_sec so the DP add-on's cost
        is visible on its own.
      - transform_time_sec (client): time spent on the post-noise
        top-k masking + quantization + permutation pass.
      - reconstruct_time_sec (server): time spent on the matching
        unpermute + dequantize pass across all clients' tensors (this
        does NOT remove the DP noise -- only quantization/permutation
        are inverted).

    DP-SPECIFIC (NEW) -- reported per round and summarized at the end:
      - dp_clip_norm, dp_epsilon, dp_delta: the configured budget.
      - dp_scale: the fraction the global selected-value vector had to
        be shrunk by to satisfy the clip norm (1.0 = no clipping needed
        this round).
      - dp_sigma: the Gaussian noise std actually used, calibrated from
        (dp_clip_norm, dp_epsilon, dp_delta).
      - dp_k_total: how many coordinates were actually noised this
        round (sum of per-tensor top-k counts).
      - dp_signal_mean_abs / dp_snr: mean absolute value of the
        selected values pre-noise, and the resulting signal-to-noise
        ratio (dp_signal_mean_abs / dp_sigma) -- the number that shows
        whether sparsification is buying a better noise/signal trade
        at a given epsilon.

    All of these are printed per round and accumulated into running
    totals, with a final summary at the last round -- mirroring the
    mask-timing / delta-norm diagnostics already added to secagg_topk.py,
    so the two scripts' overhead numbers are directly comparable.
"""

import flwr as fl
import torch
import numpy as np
import json
import time
import warnings
import zlib
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


def deterministic_hash(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


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

    @staticmethod
    def compute_quant_params(x: np.ndarray):
        x_min, x_max = float(x.min()), float(x.max())
        if x_max == x_min:
            return 1.0, x_min
        return x_max - x_min, x_min

    @staticmethod
    def quantize_with_params(x: np.ndarray, scale: float, zmin: float, num_bits: int = 8) -> np.ndarray:
        if num_bits >= 32:
            return x.astype(np.float32)
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        x_scaled = (x - zmin) / step
        floor = np.floor(x_scaled)
        prob = np.clip(x_scaled - floor, 0.0, 1.0)
        rnd = np.random.rand(*x.shape)
        x_q = floor + (rnd < prob)
        x_q = np.clip(x_q, 0, qmax)
        return x_q.astype(np.float32)

    @staticmethod
    def dequantize_with_params(x_q: np.ndarray, scale: float, zmin: float, num_bits: int = 8) -> np.ndarray:
        if num_bits >= 32:
            return x_q
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        return x_q * step + zmin

    @staticmethod
    def permute_array(x: np.ndarray, seed: int):
        rng = np.random.RandomState(seed % (2 ** 31 - 1))
        perm = rng.permutation(x.size)
        return x[perm]

    @staticmethod
    def unpermute_array(x: np.ndarray, seed: int, size: int):
        rng = np.random.RandomState(seed % (2 ** 31 - 1))
        perm = rng.permutation(size)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(size)
        return x[inv]

    # ---------------- NEW: DP helpers ----------------
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

        This is the standard textbook calibration (Dwork & Roth). It is a
        sufficient (not necessarily tight) bound; a tighter analytic
        Gaussian mechanism (Balle & Wang, 2018) could be substituted here
        without changing anything else in this pipeline.
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("epsilon must be > 0 and delta must be in (0, 1).")
        return float(clip_norm * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)

    @staticmethod
    def add_gaussian_noise(v: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return v
        return v + np.random.normal(loc=0.0, scale=sigma, size=v.shape).astype(v.dtype)

# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
NUM_CLIENTS = 3
LOCAL_EPOCHS = 5
NUM_ROUNDS = 40

# ---- privacy strategy knobs ----
USE_PRIVACY = True
PRIVACY_KEEP_RATIO = 0.3     # fraction of each tensor's elements transmitted
PRIVACY_QUANT_BITS = 8       # bits for stochastic quantization

# ---- DP add-on knobs (SAPM+DP hybrid) ----
USE_DP_NOISE = True          # master switch for the new noise mechanism;
                              # False reproduces the original SAPM script.
PRIVACY_DP_EPSILON = 8.0     # PER-ROUND epsilon (see module docstring)
PRIVACY_DP_DELTA = 1e-5      # PER-ROUND delta
PRIVACY_DP_CLIP_NORM = 1.0   # L2 clip norm C on the vector of selected
                              # (top-k) values across the whole client
                              # update, per round.

# ---- overhead-accounting knobs ----
INDEX_BYTES_PER_ELEMENT = 4  # bytes needed per sparse-encoded index (int32);
                              # used only for the comm_sparse_encoded_bytes
                              # estimate, not for the actual bytes sent.

print(f"Using device: {DEVICE}")
print(f"Privacy strategy: SAPM | enabled={USE_PRIVACY} | keep_ratio={PRIVACY_KEEP_RATIO} | quant_bits={PRIVACY_QUANT_BITS}")
print(f"DP add-on: enabled={USE_DP_NOISE} | epsilon(per-round)={PRIVACY_DP_EPSILON} | "
      f"delta(per-round)={PRIVACY_DP_DELTA} | clip_norm={PRIVACY_DP_CLIP_NORM}")

# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    test_dataset = IMUDataset(test_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset

def split_train_data(train_dataset, num_clients=NUM_CLIENTS, save_file="client_split.json", seed=42):
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

        labels = []
        for idx in indices[start:end]:
            sample = train_dataset[idx]
            label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            labels.append(label)

        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))

        print(f"Client {i} → {len(subset)} samples | Label distribution: {dist}")

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
        # Used only for initial global-model bootstrap by Flower.
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
        quant_bits = fit_config.get("privacy_quant_bits", PRIVACY_QUANT_BITS)
        round_seed = fit_config.get("privacy_seed", 0)

        use_dp_noise = fit_config.get("use_dp_noise", USE_DP_NOISE) and use_privacy
        dp_epsilon = fit_config.get("privacy_dp_epsilon", PRIVACY_DP_EPSILON)
        dp_delta = fit_config.get("privacy_dp_delta", PRIVACY_DP_DELTA)
        dp_clip_norm = fit_config.get("privacy_dp_clip_norm", PRIVACY_DP_CLIP_NORM)

        self.model.train()
        total_loss = 0.0

        # Fisher (diagonal) sensitivity accumulator, one per floating-point param
        fisher_accum = {
            name: torch.zeros_like(p)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        n_grad_steps = 0
        fisher_time_sec = 0.0  # wall-clock time spent ONLY on the Fisher
                                # accumulation itself (the "+= grad**2"
                                # step below), isolated from the rest of
                                # each training step so this overhead is
                                # visible on its own, not blended into
                                # total_loss/training time.

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

        # ---- Overhead accounting (see module docstring) ----
        comm_no_privacy_bytes = 0     # plain dense float32 baseline

        # ===== PASS 1: compute deltas, fisher-based top-k masks =====
        # For tensors under privacy, we defer masking/noising/quantizing
        # until we've seen every tensor, because the DP clip+noise step
        # is applied to the GLOBAL vector of selected values across the
        # whole model update (that's the sensitivity bound the noise is
        # calibrated against), not tensor-by-tensor.
        plain_names = []          # tensors NOT under privacy (pass-through)
        plain_arrays = {}
        priv_names = []           # tensors under privacy
        priv_delta_flat = {}      # name -> flat float32 delta
        priv_mask = {}            # name -> bool mask (top-k selection)
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

        # ===== PASS 2 (NEW): global clip + Gaussian noise over the =====
        # ===== concatenated vector of top-k selected values =====
        dp_time_sec = 0.0
        dp_scale = 1.0
        dp_sigma = 0.0
        dp_k_total = 0
        dp_signal_mean_abs = 0.0

        noised_selected = {}  # name -> noised values, same length as mask.sum()

        if priv_names:
            _dp_t0 = time.perf_counter()

            per_name_selected = {name: priv_delta_flat[name][priv_mask[name]] for name in priv_names}
            lengths = [per_name_selected[name].size for name in priv_names]
            dp_k_total = int(sum(lengths))

            if use_dp_noise and dp_k_total > 0:
                global_vec = np.concatenate([per_name_selected[name] for name in priv_names]).astype(np.float32)
                dp_signal_mean_abs = float(np.mean(np.abs(global_vec))) if global_vec.size else 0.0

                clipped_vec, dp_scale = pu.clip_l2(global_vec, dp_clip_norm)
                dp_sigma = pu.gaussian_mechanism_sigma(dp_clip_norm, dp_epsilon, dp_delta)
                noised_vec = pu.add_gaussian_noise(clipped_vec, dp_sigma).astype(np.float32)

                offset = 0
                for name, length in zip(priv_names, lengths):
                    noised_selected[name] = noised_vec[offset:offset + length]
                    offset += length
            else:
                # DP disabled (or nothing selected): pass the selected
                # values through unchanged -- behaves like plain SAPM.
                for name in priv_names:
                    noised_selected[name] = per_name_selected[name]

            dp_time_sec = time.perf_counter() - _dp_t0

        # ===== PASS 3: scatter noised values back, quantize, permute =====
        out_arrays = []
        meta = []
        nz_total, elem_total = 0, 0
        comm_dense_bytes = 0
        comm_sparse_encoded_bytes = 0
        transform_time_sec = 0.0

        for name in plain_names:
            arr = plain_arrays[name]
            out_arrays.append(arr)
            meta.append([1.0, 0.0, False])  # scale, zmin, quantized?
            nz_total += np.count_nonzero(arr)
            elem_total += arr.size
            comm_dense_bytes += arr.nbytes
            comm_sparse_encoded_bytes += arr.nbytes

        for name in priv_names:
            mask = priv_mask[name]
            shape = priv_shape[name]
            delta_flat = priv_delta_flat[name]

            _t0 = time.perf_counter()
            sparse_delta = np.zeros_like(delta_flat)
            sparse_delta[mask] = noised_selected[name]

            scale, zmin = pu.compute_quant_params(delta_flat)
            q = pu.quantize_with_params(sparse_delta, scale, zmin, quant_bits)
            permuted = pu.permute_array(q, seed=round_seed * 100003 + deterministic_hash(name) % 97)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(permuted.reshape(shape).astype(np.float32))
            meta.append([float(scale), float(zmin), True])

            nz_count = int(np.count_nonzero(mask))
            nz_total += np.count_nonzero(sparse_delta)
            elem_total += sparse_delta.size

            # AS ACTUALLY SENT: `permuted` is a full D-length float32 array
            # (zeros included) -- Flower serializes and transmits all of
            # it, so keep_ratio buys NO reduction in bytes today.
            comm_dense_bytes += permuted.astype(np.float32).nbytes
            # AS A SPARSE ENCODING WOULD COST: only the nz_count nonzero
            # entries, each as (index + quantized value).
            value_bytes = max(1, -(-quant_bits // 8))  # ceil(bits/8)
            comm_sparse_encoded_bytes += nz_count * (INDEX_BYTES_PER_ELEMENT + value_bytes)

        dp_snr = (dp_signal_mean_abs / dp_sigma) if dp_sigma > 0 else float("inf")

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "nonzero_ratio": float(nz_total / max(1, elem_total)),
            "privacy_meta": json.dumps(meta),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_sparse_encoded_bytes": comm_sparse_encoded_bytes,
            "comm_no_privacy_bytes": comm_no_privacy_bytes,
            "fisher_time_sec": fisher_time_sec,
            "transform_time_sec": transform_time_sec,
            # ---- NEW: DP diagnostics ----
            "dp_time_sec": dp_time_sec,
            "dp_clip_norm": dp_clip_norm,
            "dp_epsilon": dp_epsilon,
            "dp_delta": dp_delta,
            "dp_scale": dp_scale,
            "dp_sigma": dp_sigma,
            "dp_k_total": dp_k_total,
            "dp_signal_mean_abs": dp_signal_mean_abs,
            "dp_snr": dp_snr,
        }
        return out_arrays, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, eval_config):
        self.set_parameters(parameters)
        self.model.eval()
        all_preds = []
        all_labels = []

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
                 privacy_quant_bits=PRIVACY_QUANT_BITS,
                 use_dp_noise=USE_DP_NOISE,
                 privacy_dp_epsilon=PRIVACY_DP_EPSILON,
                 privacy_dp_delta=PRIVACY_DP_DELTA,
                 privacy_dp_clip_norm=PRIVACY_DP_CLIP_NORM,
                 **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_privacy = use_privacy
        self.privacy_keep_ratio = privacy_keep_ratio
        self.privacy_quant_bits = privacy_quant_bits
        self.use_dp_noise = use_dp_noise
        self.privacy_dp_epsilon = privacy_dp_epsilon
        self.privacy_dp_delta = privacy_dp_delta
        self.privacy_dp_clip_norm = privacy_dp_clip_norm

        # Running totals across the whole run, for the final summary.
        self.total_comm_dense_bytes = 0
        self.total_comm_sparse_encoded_bytes = 0
        self.total_comm_no_privacy_bytes = 0
        self.total_fisher_time_sec = 0.0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0
        self.total_dp_time_sec = 0.0
        self.dp_sigma_history = []
        self.dp_snr_history = []
        self.dp_k_total_history = []

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        # Same seed broadcast to every client this round: server (and only
        # the server) can invert every client's permutation deterministically.
        # The DP noise itself is drawn independently per client and is NOT
        # inverted by the server -- only quantization/permutation are.
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_privacy"] = self.use_privacy
            fit_ins.config["privacy_keep_ratio"] = self.privacy_keep_ratio
            fit_ins.config["privacy_quant_bits"] = self.privacy_quant_bits
            fit_ins.config["privacy_seed"] = server_round
            fit_ins.config["use_dp_noise"] = self.use_dp_noise
            fit_ins.config["privacy_dp_epsilon"] = self.privacy_dp_epsilon
            fit_ins.config["privacy_dp_delta"] = self.privacy_dp_delta
            fit_ins.config["privacy_dp_clip_norm"] = self.privacy_dp_clip_norm
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

        # ---- Overhead accounting (see module docstring) ----
        round_comm_dense_bytes = 0
        round_comm_sparse_encoded_bytes = 0
        round_comm_no_privacy_bytes = 0
        round_fisher_time_sec = []
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0  # server-side unpermute+dequantize

        # ---- NEW: DP accounting ----
        round_dp_time_sec = []
        round_dp_sigma = []
        round_dp_snr = []
        round_dp_k_total = []
        round_dp_scale = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("privacy_meta", "[]"))
            nz_ratios.append(fit_res.metrics.get("nonzero_ratio", 1.0))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_sparse_encoded_bytes += fit_res.metrics.get("comm_sparse_encoded_bytes", 0)
            round_comm_no_privacy_bytes += fit_res.metrics.get("comm_no_privacy_bytes", 0)
            round_fisher_time_sec.append(fit_res.metrics.get("fisher_time_sec", 0.0))
            round_transform_time_sec.append(fit_res.metrics.get("transform_time_sec", 0.0))

            round_dp_time_sec.append(fit_res.metrics.get("dp_time_sec", 0.0))
            round_dp_sigma.append(fit_res.metrics.get("dp_sigma", 0.0))
            round_dp_snr.append(fit_res.metrics.get("dp_snr", 0.0))
            round_dp_k_total.append(fit_res.metrics.get("dp_k_total", 0))
            round_dp_scale.append(fit_res.metrics.get("dp_scale", 1.0))

            _recon_start = time.perf_counter()
            for i, (k, arr) in enumerate(zip(keys, arrays)):
                scale, zmin, quantized = meta[i] if i < len(meta) else (1.0, 0.0, False)
                flat = arr.reshape(-1)

                if quantized:
                    seed = server_round * 100003 + deterministic_hash(k) % 97
                    unpermuted = pu.unpermute_array(flat, seed=seed, size=flat.size)
                    reconstructed = pu.dequantize_with_params(unpermuted, scale, zmin, self.privacy_quant_bits)
                else:
                    reconstructed = flat

                weighted_deltas[k] += reconstructed.reshape(global_state[k].shape).astype(np.float64) * num_examples
            round_reconstruct_time_sec += time.perf_counter() - _recon_start

            total_examples += num_examples

        new_state = {}
        for k in keys:
            avg_delta = weighted_deltas[k] / max(1, total_examples)
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype, device=global_state[k].device)

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_nz = float(np.mean(nz_ratios)) if nz_ratios else 1.0

        # Update running totals.
        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_sparse_encoded_bytes += round_comm_sparse_encoded_bytes
        self.total_comm_no_privacy_bytes += round_comm_no_privacy_bytes
        avg_fisher_time = float(np.mean(round_fisher_time_sec)) if round_fisher_time_sec else 0.0
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_fisher_time_sec += avg_fisher_time
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        avg_dp_time = float(np.mean(round_dp_time_sec)) if round_dp_time_sec else 0.0
        avg_dp_sigma = float(np.mean(round_dp_sigma)) if round_dp_sigma else 0.0
        finite_snrs = [s for s in round_dp_snr if np.isfinite(s)]
        avg_dp_snr = float(np.mean(finite_snrs)) if finite_snrs else float("inf")
        avg_dp_k_total = float(np.mean(round_dp_k_total)) if round_dp_k_total else 0.0
        avg_dp_scale = float(np.mean(round_dp_scale)) if round_dp_scale else 1.0
        self.total_dp_time_sec += avg_dp_time
        self.dp_sigma_history.append(avg_dp_sigma)
        self.dp_snr_history.append(avg_dp_snr)
        self.dp_k_total_history.append(avg_dp_k_total)

        compression_vs_no_privacy = (
            round_comm_sparse_encoded_bytes / round_comm_no_privacy_bytes
            if round_comm_no_privacy_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | Avg transmitted nonzero ratio: {avg_nz:.3f}")
        print(f"  [comm] as-sent (dense): {round_comm_dense_bytes/1e6:.3f} MB | "
              f"sparse-encoded (achievable): {round_comm_sparse_encoded_bytes/1e6:.3f} MB "
              f"({compression_vs_no_privacy*100:.1f}% of no-privacy baseline) | "
              f"no-privacy baseline: {round_comm_no_privacy_bytes/1e6:.3f} MB")
        print(f"  [compute] avg client fisher_time: {avg_fisher_time*1000:.2f}ms | "
              f"avg client dp_time: {avg_dp_time*1000:.2f}ms | "
              f"avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")
        print(f"  [dp] k_total={avg_dp_k_total:.0f} | clip_scale={avg_dp_scale:.4f} | "
              f"sigma={avg_dp_sigma:.6g} | snr={avg_dp_snr:.3f} "
              f"(epsilon={self.privacy_dp_epsilon}, delta={self.privacy_dp_delta}, "
              f"clip_norm={self.privacy_dp_clip_norm})")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model.pth")

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
        }

    def print_overhead_summary(self):
        print("\n========== OVERHEAD SUMMARY (SAPM, cumulative over the run) ==========")
        print(f"Total communication AS ACTUALLY SENT (dense arrays)      : "
              f"{self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication IF SPARSE-ENCODED (achievable target): "
              f"{self.total_comm_sparse_encoded_bytes/1e6:.2f} MB")
        print(f"Total communication with NO privacy (dense, keep_ratio=1): "
              f"{self.total_comm_no_privacy_bytes/1e6:.2f} MB")
        if self.total_comm_no_privacy_bytes:
            print(f"  -> as-sent is {self.total_comm_dense_bytes/self.total_comm_no_privacy_bytes*100:.1f}% "
                  f"of no-privacy baseline (i.e. NO bandwidth savings today -- see module docstring)")
            print(f"  -> sparse-encoded WOULD BE {self.total_comm_sparse_encoded_bytes/self.total_comm_no_privacy_bytes*100:.1f}% "
                  f"of no-privacy baseline if implemented (this is the number to compare against secagg_topk.py's "
                  f"real K-length transmissions)")
        print(f"Total client-side fisher accumulation time (avg client, summed over rounds): "
              f"{self.total_fisher_time_sec:.2f}s")
        print(f"Total client-side DP clip+noise time (avg client, summed over rounds)      : "
              f"{self.total_dp_time_sec:.2f}s")
        print(f"Total client-side transform time (mask+quant+permute, avg client, summed)  : "
              f"{self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (unpermute+dequantize, summed over rounds): "
              f"{self.total_reconstruct_time_sec:.2f}s")

        print("\n---- DP (SAPM+DP hybrid) summary ----")
        print(f"Per-round budget used: epsilon={self.privacy_dp_epsilon}, delta={self.privacy_dp_delta}, "
              f"clip_norm={self.privacy_dp_clip_norm}")
        naive_total_epsilon = self.privacy_dp_epsilon * NUM_ROUNDS
        print(f"Naive BASIC-COMPOSITION total epsilon over {NUM_ROUNDS} rounds: {naive_total_epsilon:.2f} "
              f"(a crude, conservative upper bound -- use an RDP/moments accountant for a tight bound; "
              f"advanced composition or subsampling amplification would give a materially smaller number)")
        if self.dp_sigma_history:
            print(f"Noise std sigma used per round: {self.dp_sigma_history[0]:.6g} "
                  f"(constant across rounds -- depends only on clip_norm/epsilon/delta, NOT on k or D)")
        if self.dp_k_total_history:
            avg_k = float(np.mean(self.dp_k_total_history))
            print(f"Average number of coordinates actually noised per round (k_total): {avg_k:.0f} "
                  f"(vs. full model size if keep_ratio were 1.0)")
        finite_snrs = [s for s in self.dp_snr_history if np.isfinite(s)]
        if finite_snrs:
            print(f"Average empirical signal-to-noise ratio (mean|selected value| / sigma): "
                  f"{np.mean(finite_snrs):.3f}")
            print("  -> Re-run with PRIVACY_KEEP_RATIO=1.0 (noise spread over the full dense delta, the naive "
                  "'full DP' baseline) and compare this SNR against the sparsified run above: for the same "
                  "(epsilon, delta, clip_norm), a smaller keep_ratio concentrates the clip budget onto fewer, "
                  "higher-Fisher-sensitivity coordinates, which typically raises this SNR -- i.e. better "
                  "empirical protection at a comparable or better signal quality, or equivalently room to "
                  "tighten epsilon further at the same signal quality.")

    def evaluate_global(self, final=False):
        self.global_model.eval()

        all_preds = []
        all_labels = []

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
        plt.title("Final Confusion Matrix (SAPM+DP)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_sapm_dp.png")
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
        privacy_quant_bits=PRIVACY_QUANT_BITS,
        use_dp_noise=USE_DP_NOISE,
        privacy_dp_epsilon=PRIVACY_DP_EPSILON,
        privacy_dp_delta=PRIVACY_DP_DELTA,
        privacy_dp_clip_norm=PRIVACY_DP_CLIP_NORM,
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
    main("train.csv", "test.csv")