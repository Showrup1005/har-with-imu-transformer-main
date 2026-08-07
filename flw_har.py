"""
Federated HAR training with SAPM-Stable
(Sensitivity-Aware Private Masking + real sparse transmission +
 decaying keep_ratio schedule + Fisher-weighted local regularization).

This extends fl_train.py (SAPM) with three changes, each independently
verifiable from the printed metrics:

  1. REAL SPARSE TRANSMISSION
     The original SAPM script builds a full D-length dense array with
     (1-keep_ratio) of it zeroed and sends that whole array -- so
     keep_ratio never actually reduced bytes on the wire.
     Here, each masked tensor is sent as TWO short arrays: int32
     indices and their quantized values, both of length nz_count (the
     number of kept elements), instead of one D-length dense array.
     comm_dense_bytes below is now genuinely "what got sent" -- you
     can compare it directly against comm_no_privacy_bytes to see a
     real compression ratio, not a hypothetical one.

  2. DECAYING KEEP_RATIO SCHEDULE
     keep_ratio starts high (more signal while the model is still
     learning) and decays toward a lower floor over the course of
     training (more compression once the model is closer to
     converged). Cosine schedule between PRIVACY_KEEP_RATIO_START and
     PRIVACY_KEEP_RATIO_END, set server-side and broadcast per round.

  3. FISHER-WEIGHTED LOCAL REGULARIZATION (EWC-lite)
     Local loss becomes CE + EWC_LAMBDA * sum_i F_i * (theta_i -
     theta_global_i)^2, using the SAME running Fisher accumulator SAPM
     already computes -- no extra backward pass, no extra tensors.
     F_i is the running (not-yet-finalized) per-parameter Fisher
     estimate at that point in local training. This discourages the
     client from moving far, on the parameters the model is most
     sensitive to, away from the round's starting point -- which
     matters more as keep_ratio drops and less of the update survives
     to be transmitted, so the client should spend its "budget" of
     transmitted signal on directions that matter and not wander on
     the rest.
     NOTE / approximation: the Fisher accumulator is updated from the
     COMBINED loss gradient (CE + regularizer), not a CE-only gradient,
     to avoid a second backward pass per step. This mildly contaminates
     the Fisher signal with the regularizer's own curvature; in
     practice this effect is small for reasonable EWC_LAMBDA and is a
     deliberate simplicity/cost tradeoff -- flagged here rather than
     hidden.

Everything else (model, dataset, overall FL loop, quantization,
permutation) is unchanged from fl_train.py.
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


# ====================== PRIVACY (SAPM-Stable) HELPERS ======================
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
    def permute_pair(idx: np.ndarray, val: np.ndarray, seed: int):
        # Jointly permutes the compact (indices, values) pair -- length
        # is nz_count now, not the full tensor size.
        rng = np.random.RandomState(seed % (2 ** 31 - 1))
        perm = rng.permutation(idx.size)
        return idx[perm], val[perm]

    @staticmethod
    def unpermute_pair(idx: np.ndarray, val: np.ndarray, seed: int):
        rng = np.random.RandomState(seed % (2 ** 31 - 1))
        perm = rng.permutation(idx.size)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(idx.size)
        return idx[inv], val[inv]

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

USE_PRIVACY = True
PRIVACY_KEEP_RATIO_START = 0.6   # round 1
PRIVACY_KEEP_RATIO_END = 0.15    # final round; cosine decay between the two
PRIVACY_QUANT_BITS = 8
EWC_LAMBDA = 0.01                # weight on the Fisher-weighted regularizer

INDEX_BYTES_PER_ELEMENT = 4  # int32 index actually transmitted now

print(f"Using device: {DEVICE}")
print(f"Privacy strategy: SAPM-Stable | enabled={USE_PRIVACY} | "
      f"keep_ratio {PRIVACY_KEEP_RATIO_START}->{PRIVACY_KEEP_RATIO_END} (cosine) | "
      f"quant_bits={PRIVACY_QUANT_BITS} | ewc_lambda={EWC_LAMBDA}")

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
        keep_ratio = fit_config.get("privacy_keep_ratio", PRIVACY_KEEP_RATIO_START)
        quant_bits = fit_config.get("privacy_quant_bits", PRIVACY_QUANT_BITS)
        round_seed = fit_config.get("privacy_seed", 0)
        ewc_lambda = fit_config.get("ewc_lambda", EWC_LAMBDA)

        self.model.train()
        total_loss = 0.0
        total_reg_loss = 0.0

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
                ce_loss = self.criterion(output, label)

                # ---- Fisher-weighted regularizer (uses running Fisher
                # estimate so far this round; zero on the very first
                # step since there's no estimate yet) ----
                reg_loss = torch.zeros((), device=DEVICE)
                if use_privacy and ewc_lambda > 0 and n_grad_steps > 0:
                    for name, p in self.model.named_parameters():
                        if name in fisher_accum:
                            f_running = (fisher_accum[name] / n_grad_steps).detach()
                            reg_loss = reg_loss + (f_running * (p - old_state[name]) ** 2).sum()

                loss = ce_loss + ewc_lambda * reg_loss
                loss.backward()

                _fisher_start = time.perf_counter()
                for name, p in self.model.named_parameters():
                    if p.grad is not None and name in fisher_accum:
                        fisher_accum[name] += p.grad.detach() ** 2
                fisher_time_sec += time.perf_counter() - _fisher_start
                n_grad_steps += 1

                self.optimizer.step()
                total_loss += ce_loss.item()
                total_reg_loss += float(reg_loss.detach().item())

        for k in fisher_accum:
            fisher_accum[k] /= max(1, n_grad_steps)

        new_state = self.model.state_dict()

        out_arrays = []
        meta = []
        nz_total, elem_total = 0, 0

        comm_dense_bytes = 0          # now genuinely what's transmitted
        comm_sparse_encoded_bytes = 0  # bit-packed lower bound (still hypothetical re: sub-byte packing)
        comm_no_privacy_bytes = 0
        transform_time_sec = 0.0

        for name, new_val in new_state.items():
            old_val = old_state[name]
            delta = (new_val - old_val).cpu().numpy()
            comm_no_privacy_bytes += delta.astype(np.float32).nbytes

            if not use_privacy or name not in fisher_accum:
                out_arrays.append(delta.astype(np.float32))
                meta.append({"quantized": False, "sparse": False,
                             "shape": list(delta.shape), "size": int(delta.size)})
                nz_total += np.count_nonzero(delta)
                elem_total += delta.size
                comm_dense_bytes += delta.astype(np.float32).nbytes
                comm_sparse_encoded_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            fisher_flat = fisher_accum[name].cpu().numpy().reshape(-1)
            delta_flat = delta.reshape(-1).astype(np.float32)

            mask = pu.compute_topk_mask(fisher_flat, keep_ratio)
            idx = np.nonzero(mask)[0].astype(np.int32)

            scale, zmin = pu.compute_quant_params(delta_flat)
            q_vals = pu.quantize_with_params(delta_flat[idx], scale, zmin, quant_bits).astype(np.float32)

            seed = round_seed * 100003 + deterministic_hash(name) % 97
            perm_idx, perm_vals = pu.permute_pair(idx, q_vals, seed=seed)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(perm_idx)
            out_arrays.append(perm_vals)
            meta.append({"quantized": True, "sparse": True,
                         "shape": list(delta.shape), "size": int(delta.size),
                         "scale": float(scale), "zmin": float(zmin),
                         "nz_count": int(idx.size)})

            nz_total += idx.size
            elem_total += delta.size

            comm_dense_bytes += perm_idx.nbytes + perm_vals.astype(np.float32).nbytes
            value_bytes = max(1, -(-quant_bits // 8))
            comm_sparse_encoded_bytes += idx.size * (INDEX_BYTES_PER_ELEMENT + value_bytes)

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "avg_reg_loss": total_reg_loss / len(self.train_loader),
            "nonzero_ratio": float(nz_total / max(1, elem_total)),
            "privacy_meta": json.dumps(meta),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_sparse_encoded_bytes": comm_sparse_encoded_bytes,
            "comm_no_privacy_bytes": comm_no_privacy_bytes,
            "fisher_time_sec": fisher_time_sec,
            "transform_time_sec": transform_time_sec,
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
                 keep_ratio_start=PRIVACY_KEEP_RATIO_START,
                 keep_ratio_end=PRIVACY_KEEP_RATIO_END,
                 privacy_quant_bits=PRIVACY_QUANT_BITS,
                 ewc_lambda=EWC_LAMBDA, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_privacy = use_privacy
        self.keep_ratio_start = keep_ratio_start
        self.keep_ratio_end = keep_ratio_end
        self.privacy_quant_bits = privacy_quant_bits
        self.ewc_lambda = ewc_lambda

        self.total_comm_dense_bytes = 0
        self.total_comm_sparse_encoded_bytes = 0
        self.total_comm_no_privacy_bytes = 0
        self.total_fisher_time_sec = 0.0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

    def _keep_ratio_for_round(self, server_round):
        frac = (server_round - 1) / max(1, NUM_ROUNDS - 1)
        cos = 0.5 * (1 + np.cos(np.pi * frac))  # 1 -> 0 over the run
        ratio = self.keep_ratio_end + (self.keep_ratio_start - self.keep_ratio_end) * cos
        return float(ratio)

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        keep_ratio = self._keep_ratio_for_round(server_round)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_privacy"] = self.use_privacy
            fit_ins.config["privacy_keep_ratio"] = keep_ratio
            fit_ins.config["privacy_quant_bits"] = self.privacy_quant_bits
            fit_ins.config["privacy_seed"] = server_round
            fit_ins.config["ewc_lambda"] = self.ewc_lambda
        self._current_keep_ratio = keep_ratio
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
        reg_losses = []

        round_comm_dense_bytes = 0
        round_comm_sparse_encoded_bytes = 0
        round_comm_no_privacy_bytes = 0
        round_fisher_time_sec = []
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("privacy_meta", "[]"))
            nz_ratios.append(fit_res.metrics.get("nonzero_ratio", 1.0))
            reg_losses.append(fit_res.metrics.get("avg_reg_loss", 0.0))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_sparse_encoded_bytes += fit_res.metrics.get("comm_sparse_encoded_bytes", 0)
            round_comm_no_privacy_bytes += fit_res.metrics.get("comm_no_privacy_bytes", 0)
            round_fisher_time_sec.append(fit_res.metrics.get("fisher_time_sec", 0.0))
            round_transform_time_sec.append(fit_res.metrics.get("transform_time_sec", 0.0))

            _recon_start = time.perf_counter()
            cursor = 0
            for k, m in zip(keys, meta):
                shape = tuple(m["shape"])
                size = m["size"]

                if not m["sparse"]:
                    arr = arrays[cursor]; cursor += 1
                    reconstructed = arr.reshape(shape)
                else:
                    idx_arr = arrays[cursor]; cursor += 1
                    val_arr = arrays[cursor]; cursor += 1
                    seed = server_round * 100003 + deterministic_hash(k) % 97
                    idx_unperm, val_unperm = pu.unpermute_pair(idx_arr, val_arr, seed=seed)
                    dequant = pu.dequantize_with_params(val_unperm, m["scale"], m["zmin"], self.privacy_quant_bits)
                    dense = np.zeros(size, dtype=np.float32)
                    dense[idx_unperm] = dequant
                    reconstructed = dense.reshape(shape)

                weighted_deltas[k] += reconstructed.astype(np.float64) * num_examples
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
        avg_reg = float(np.mean(reg_losses)) if reg_losses else 0.0

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_sparse_encoded_bytes += round_comm_sparse_encoded_bytes
        self.total_comm_no_privacy_bytes += round_comm_no_privacy_bytes
        avg_fisher_time = float(np.mean(round_fisher_time_sec)) if round_fisher_time_sec else 0.0
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_fisher_time_sec += avg_fisher_time
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        compression_vs_no_privacy = (
            round_comm_dense_bytes / round_comm_no_privacy_bytes
            if round_comm_no_privacy_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | keep_ratio={self._current_keep_ratio:.3f} | "
              f"Avg transmitted nonzero ratio: {avg_nz:.3f} | avg_reg_loss: {avg_reg:.5f}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_no_privacy*100:.1f}% of no-privacy baseline: {round_comm_no_privacy_bytes/1e6:.3f} MB) | "
              f"bit-packed lower bound: {round_comm_sparse_encoded_bytes/1e6:.3f} MB")
        print(f"  [compute] avg client fisher_time: {avg_fisher_time*1000:.2f}ms | "
              f"avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_sapm_stable.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.print_overhead_summary()

        return aggregated_params, {
            "accuracy": acc,
            "avg_nonzero_ratio": avg_nz,
            "comm_dense_bytes": round_comm_dense_bytes,
        }

    def print_overhead_summary(self):
        print("\n========== OVERHEAD SUMMARY (SAPM-Stable, cumulative over the run) ==========")
        print(f"Total communication ACTUALLY SENT (sparse, real)  : {self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication with NO privacy (dense baseline): {self.total_comm_no_privacy_bytes/1e6:.2f} MB")
        if self.total_comm_no_privacy_bytes:
            print(f"  -> real compression achieved: {self.total_comm_dense_bytes/self.total_comm_no_privacy_bytes*100:.1f}% "
                  f"of no-privacy baseline")
            print(f"  -> bit-packed lower bound would be {self.total_comm_sparse_encoded_bytes/self.total_comm_no_privacy_bytes*100:.1f}% "
                  f"of no-privacy baseline (further headroom if values are bit-packed instead of stored as float32)")
        print(f"Total client-side fisher accumulation time (avg client, summed over rounds): {self.total_fisher_time_sec:.2f}s")
        print(f"Total client-side transform time (mask+quant+permute, avg client, summed)  : {self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (unpermute+dequantize+scatter, summed)   : {self.total_reconstruct_time_sec:.2f}s")
        print(f"Best accuracy achieved: {self.best_acc:.4f}")

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
        plt.title("Final Confusion Matrix (SAPM-Stable)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_sapm_stable.png")
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
        keep_ratio_start=PRIVACY_KEEP_RATIO_START,
        keep_ratio_end=PRIVACY_KEEP_RATIO_END,
        privacy_quant_bits=PRIVACY_QUANT_BITS,
        ewc_lambda=EWC_LAMBDA,
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