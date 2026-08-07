"""
Federated HAR training with SAC (Sensitivity-Aware Compression).

This is a compression-focused variant of the SAPM-Stable script. The
privacy-only mechanism (index/value permutation, which added compute
cost and zero compression benefit -- it only obscured *which*
positions were transmitted) has been removed. What's kept and
strengthened is everything that trades bytes for accuracy in a
principled way:

  1. REAL SPARSE + REAL NARROW-DTYPE TRANSMISSION
     Each masked tensor is sent as two short arrays: indices and
     quantized values, both of length nz_count. Two fixes vs. the
     prior script:
       - Values are cast to their true quantized width (uint8 for
         8-bit quant, uint16 for up to 16-bit) instead of being
         stored back as float32. This alone is a ~4x reduction on
         the value stream for 8-bit quantization.
       - Indices use the narrowest integer dtype that can address
         the tensor (uint16 below 65536 elements, else uint32)
         instead of always int32.
     comm_dense_bytes now reflects what's *actually* on the wire in
     these narrow dtypes, so it should land much closer to the old
     "bit-packed lower bound" line than before.

  2. DECAYING KEEP_RATIO SCHEDULE
     keep_ratio starts high (more signal while the model is still
     learning) and decays toward a lower floor over training (more
     compression once the model is closer to converged). Cosine
     schedule between COMPRESS_KEEP_RATIO_START/END, set server-side
     and broadcast per round.

  3. FISHER-WEIGHTED STABILITY REGULARIZATION
     Local loss becomes CE + STABILITY_LAMBDA * sum_i F_i * (theta_i
     - theta_global_i)^2, using the running Fisher accumulator the
     mask selection already computes -- no extra backward pass. This
     keeps clients from drifting on the parameters the model is most
     sensitive to, which matters more as keep_ratio drops and less of
     the update survives transmission -- so the client should spend
     its transmitted "budget" on directions that matter and hold
     still on the rest. This is an accuracy-preservation mechanism,
     not a privacy mechanism, and is kept for exactly that reason.
     NOTE / approximation: the Fisher accumulator is updated from the
     COMBINED loss gradient (CE + regularizer), not a CE-only
     gradient, to avoid a second backward pass per step. This mildly
     contaminates the Fisher signal with the regularizer's own
     curvature; in practice this effect is small for reasonable
     STABILITY_LAMBDA and is a deliberate simplicity/cost tradeoff.

Everything else (model, dataset, overall FL loop) is unchanged.
"""

import flwr as fl
import torch
import numpy as np
import json
import time
import warnings
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# ====================== COMPRESSION HELPERS ======================
class cu:
    @staticmethod
    def compute_topk_mask(fisher_flat: np.ndarray, keep_ratio: float) -> np.ndarray:
        """Keep the top-`keep_ratio` fraction of elements by Fisher
        sensitivity. This is the actual compression lever: fewer kept
        elements = fewer bytes on the wire."""
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
        """Stochastic-rounding quantization: unbiased in expectation,
        which matters for accuracy once keep_ratio and quant_bits are
        both aggressive. Returns integer codes in [0, 2**num_bits-1]
        as float32; caller casts to the narrow output dtype."""
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
            return x_q.astype(np.float32)
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        return x_q.astype(np.float32) * step + zmin

    @staticmethod
    def index_dtype_for_size(size: int):
        """Narrowest integer dtype that can address this tensor."""
        return np.uint16 if size <= 65535 else np.uint32

    @staticmethod
    def value_dtype_for_bits(num_bits: int):
        if num_bits <= 8:
            return np.uint8
        elif num_bits <= 16:
            return np.uint16
        else:
            return np.float32


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

USE_COMPRESSION = True
COMPRESS_KEEP_RATIO_START = 0.6   # round 1
COMPRESS_KEEP_RATIO_END = 0.22    # was 0.15 -- 0.15 was still starving late-stage fine-tuning
QUANT_BITS = 8
STABILITY_LAMBDA = 0.01           # weight on the Fisher-weighted stability regularizer

# Tensors at or below this many elements (biases, LayerNorm params, the
# final classifier head) are sent dense + unquantized every round. They
# are cheap in absolute bytes no matter what, but disproportionately
# important for class boundaries -- masking/quantizing them saves
# almost nothing on the wire while directly hurting accuracy on the
# hardest, most confusable classes. Everything above this threshold
# still goes through the full mask+quantize pipeline as before.
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096

print(f"Using device: {DEVICE}")
print(f"Compression strategy: SAC | enabled={USE_COMPRESSION} | "
      f"keep_ratio {COMPRESS_KEEP_RATIO_START}->{COMPRESS_KEEP_RATIO_END} (cosine) | "
      f"quant_bits={QUANT_BITS} | stability_lambda={STABILITY_LAMBDA}")

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

        use_compression = fit_config.get("use_compression", USE_COMPRESSION)
        keep_ratio = fit_config.get("compress_keep_ratio", COMPRESS_KEEP_RATIO_START)
        quant_bits = fit_config.get("quant_bits", QUANT_BITS)
        stability_lambda = fit_config.get("stability_lambda", STABILITY_LAMBDA)

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

                # ---- Fisher-weighted stability regularizer (uses
                # running Fisher estimate so far this round; zero on
                # the very first step since there's no estimate yet) ----
                reg_loss = torch.zeros((), device=DEVICE)
                if use_compression and stability_lambda > 0 and n_grad_steps > 0:
                    for name, p in self.model.named_parameters():
                        if name in fisher_accum:
                            f_running = (fisher_accum[name] / n_grad_steps).detach()
                            reg_loss = reg_loss + (f_running * (p - old_state[name]) ** 2).sum()

                loss = ce_loss + stability_lambda * reg_loss
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

        comm_dense_bytes = 0            # actual bytes on the wire, narrow dtypes
        comm_bitpacked_bytes = 0        # sub-byte theoretical lower bound (arithmetic/bit packing)
        comm_no_compression_bytes = 0
        transform_time_sec = 0.0

        for name, new_val in new_state.items():
            old_val = old_state[name]
            delta = (new_val - old_val).cpu().numpy()
            comm_no_compression_bytes += delta.astype(np.float32).nbytes

            if (not use_compression or name not in fisher_accum
                    or delta.size <= SMALL_TENSOR_FULL_SEND_THRESHOLD):
                out_arrays.append(delta.astype(np.float32))
                meta.append({"quantized": False, "sparse": False,
                             "shape": list(delta.shape), "size": int(delta.size)})
                nz_total += delta.size  # counted as fully "kept" for the nonzero-ratio metric
                elem_total += delta.size
                comm_dense_bytes += delta.astype(np.float32).nbytes
                comm_bitpacked_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            fisher_flat = fisher_accum[name].cpu().numpy().reshape(-1)
            delta_flat = delta.reshape(-1).astype(np.float32)

            mask = cu.compute_topk_mask(fisher_flat, keep_ratio)
            idx_dtype = cu.index_dtype_for_size(delta.size)
            idx = np.nonzero(mask)[0].astype(idx_dtype)  # ascending order, kept as-is

            scale, zmin = cu.compute_quant_params(delta_flat)
            q_codes = cu.quantize_with_params(delta_flat[idx], scale, zmin, quant_bits)
            q_vals = q_codes.astype(cu.value_dtype_for_bits(quant_bits))
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(idx)
            out_arrays.append(q_vals)
            meta.append({"quantized": True, "sparse": True,
                         "shape": list(delta.shape), "size": int(delta.size),
                         "scale": float(scale), "zmin": float(zmin),
                         "nz_count": int(idx.size)})

            nz_total += idx.size
            elem_total += delta.size

            # Real bytes, in the narrow dtypes actually transmitted.
            comm_dense_bytes += idx.nbytes + q_vals.nbytes

            # Sub-byte theoretical floor: ceil(log2(size)) bits/index + quant_bits/value.
            index_bits_needed = max(1, int(np.ceil(np.log2(max(2, delta.size)))))
            comm_bitpacked_bytes += idx.size * (index_bits_needed + quant_bits) / 8.0

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "avg_reg_loss": total_reg_loss / len(self.train_loader),
            "nonzero_ratio": float(nz_total / max(1, elem_total)),
            "compression_meta": json.dumps(meta),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_bitpacked_bytes": comm_bitpacked_bytes,
            "comm_no_compression_bytes": comm_no_compression_bytes,
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
    def __init__(self, test_loader, use_compression=USE_COMPRESSION,
                 keep_ratio_start=COMPRESS_KEEP_RATIO_START,
                 keep_ratio_end=COMPRESS_KEEP_RATIO_END,
                 quant_bits=QUANT_BITS,
                 stability_lambda=STABILITY_LAMBDA, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_compression = use_compression
        self.keep_ratio_start = keep_ratio_start
        self.keep_ratio_end = keep_ratio_end
        self.quant_bits = quant_bits
        self.stability_lambda = stability_lambda

        self.total_comm_dense_bytes = 0
        self.total_comm_bitpacked_bytes = 0
        self.total_comm_no_compression_bytes = 0
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
            fit_ins.config["use_compression"] = self.use_compression
            fit_ins.config["compress_keep_ratio"] = keep_ratio
            fit_ins.config["quant_bits"] = self.quant_bits
            fit_ins.config["stability_lambda"] = self.stability_lambda
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
        round_comm_bitpacked_bytes = 0
        round_comm_no_compression_bytes = 0
        round_fisher_time_sec = []
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("compression_meta", "[]"))
            nz_ratios.append(fit_res.metrics.get("nonzero_ratio", 1.0))
            reg_losses.append(fit_res.metrics.get("avg_reg_loss", 0.0))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_bitpacked_bytes += fit_res.metrics.get("comm_bitpacked_bytes", 0)
            round_comm_no_compression_bytes += fit_res.metrics.get("comm_no_compression_bytes", 0)
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
                    dequant = cu.dequantize_with_params(val_arr, m["scale"], m["zmin"], self.quant_bits)
                    dense = np.zeros(size, dtype=np.float32)
                    dense[idx_arr] = dequant
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
        self.total_comm_bitpacked_bytes += round_comm_bitpacked_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_fisher_time = float(np.mean(round_fisher_time_sec)) if round_fisher_time_sec else 0.0
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_fisher_time_sec += avg_fisher_time
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | keep_ratio={self._current_keep_ratio:.3f} | "
              f"Avg transmitted nonzero ratio: {avg_nz:.3f} | avg_reg_loss: {avg_reg:.5f}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB) | "
              f"sub-byte bit-packed floor: {round_comm_bitpacked_bytes/1e6:.3f} MB")
        print(f"  [compute] avg client fisher_time: {avg_fisher_time*1000:.2f}ms | "
              f"avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_sac.pth")

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
        print("\n========== OVERHEAD SUMMARY (SAC, cumulative over the run) ==========")
        print(f"Total communication ACTUALLY SENT (sparse, narrow-dtype)  : {self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication with NO compression (dense baseline): {self.total_comm_no_compression_bytes/1e6:.2f} MB")
        if self.total_comm_no_compression_bytes:
            print(f"  -> real compression achieved: {self.total_comm_dense_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline")
            print(f"  -> sub-byte bit-packing floor would be {self.total_comm_bitpacked_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline (further headroom if index/value bits are packed below byte boundaries)")
        print(f"Total client-side fisher accumulation time (avg client, summed over rounds): {self.total_fisher_time_sec:.2f}s")
        print(f"Total client-side transform time (mask+quant, avg client, summed)  : {self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (dequantize+scatter, summed)   : {self.total_reconstruct_time_sec:.2f}s")
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
        plt.title("Final Confusion Matrix (SAC)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_sac.png")
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
        use_compression=USE_COMPRESSION,
        keep_ratio_start=COMPRESS_KEEP_RATIO_START,
        keep_ratio_end=COMPRESS_KEEP_RATIO_END,
        quant_bits=QUANT_BITS,
        stability_lambda=STABILITY_LAMBDA,
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