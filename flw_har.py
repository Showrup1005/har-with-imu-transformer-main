"""
Fisher-Guided Mixed-Precision Compression (FGMP)

Combines SAC's Fisher-sensitivity scoring with INT8-style dense
quantization, but changes what the Fisher score is used FOR:

  SAC:  Fisher score decides who gets sent at all (top-k). Everyone
        else is hard-zeroed -- their update is discarded entirely.
  INT8: everyone gets sent, but at the same 8-bit precision
        regardless of importance.
  FGMP: everyone gets sent (nothing is ever zeroed), but Fisher score
        decides precision: the top `high_ratio` fraction by Fisher
        sensitivity gets HIGH_BITS (8-bit), everyone else gets
        LOW_BITS (4-bit, nibble-packed two values per byte).

Per-tensor cost: high-precision positions are addressed however is
cheaper this round -- a flat bitmask (1 bit/element) when high_ratio
is large, or delta-encoded indices (cheap only when few elements need
addressing) once high_ratio drops below ~1/index_bits (~0.125 for a
1-byte delta). 
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


# ====================== MIXED-PRECISION HELPERS ======================
class mp:
    @staticmethod
    def compute_quant_params(x: np.ndarray):
        if x.size == 0:
            return 1.0, 0.0
        x_min, x_max = float(x.min()), float(x.max())
        if x_max == x_min:
            return 1.0, x_min
        return x_max - x_min, x_min

    @staticmethod
    def quantize_with_params(x: np.ndarray, scale: float, zmin: float, num_bits: int) -> np.ndarray:
        """Stochastic-rounding quantization, same as SAC's -- unbiased
        in expectation, which matters more here since the low-precision
        group only gets 4 bits."""
        if x.size == 0:
            return x.astype(np.float32)
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        x_scaled = (x - zmin) / step
        floor = np.floor(x_scaled)
        prob = np.clip(x_scaled - floor, 0.0, 1.0)
        rnd = np.random.rand(*x.shape)
        x_q = floor + (rnd < prob)
        return np.clip(x_q, 0, qmax).astype(np.float32)

    @staticmethod
    def dequantize_with_params(x_q: np.ndarray, scale: float, zmin: float, num_bits: int) -> np.ndarray:
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        return x_q.astype(np.float32) * step + zmin

    @staticmethod
    def pack_nibbles(values_0to15: np.ndarray) -> np.ndarray:
        """Two 4-bit codes per byte. Pads with a zero if odd length."""
        v = values_0to15.astype(np.uint8)
        if v.size % 2 == 1:
            v = np.concatenate([v, [0]])
        pairs = v.reshape(-1, 2)
        return ((pairs[:, 0] & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)).astype(np.uint8)

    @staticmethod
    def unpack_nibbles(packed: np.ndarray, n: int) -> np.ndarray:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        out = np.empty(packed.size * 2, dtype=np.uint8)
        out[0::2] = lo
        out[1::2] = hi
        return out[:n]

    @staticmethod
    def index_dtype_for_max_delta(max_delta: int):
        if max_delta <= 255:
            return np.uint8
        elif max_delta <= 65535:
            return np.uint16
        else:
            return np.uint32

    @staticmethod
    def encode(delta_flat: np.ndarray, fisher_flat: np.ndarray, high_ratio: float,
               high_bits: int = 8, low_bits: int = 4) -> dict:
        """Adaptive addressing: the high-precision subset's positions are
        sent either as a bitmask (1 bit/element, flat cost regardless of
        how few elements are high-precision) or as delta-encoded indices
        (cheap when few elements are high-precision, expensive when many
        are). Breakeven is high_ratio ~= 1/index_bits (~0.125 for a
        1-byte delta) -- below that, addressing directly beats marking
        every element. Whichever costs fewer real bytes for THIS tensor
        THIS round is what gets sent; nothing about what's quantized or
        dropped changes, only how positions are communicated."""
        n = delta_flat.size
        k = max(1, int(np.ceil(high_ratio * n)))
        if k >= n:
            mask = np.ones(n, dtype=bool)
        else:
            idx_high = np.argpartition(fisher_flat, -k)[-k:]
            mask = np.zeros(n, dtype=bool)
            mask[idx_high] = True

        high_vals = delta_flat[mask]
        low_vals = delta_flat[~mask]

        h_scale, h_zmin = mp.compute_quant_params(high_vals)
        l_scale, l_zmin = mp.compute_quant_params(low_vals)

        h_q = mp.quantize_with_params(high_vals, h_scale, h_zmin, high_bits).astype(np.uint8)
        l_q = mp.quantize_with_params(low_vals, l_scale, l_zmin, low_bits).astype(np.uint8)
        low_packed = mp.pack_nibbles(l_q)

        # Option A: bitmask
        mask_packed = np.packbits(mask)

        # Option B: delta-encoded ascending indices of the high group
        high_idx_sorted = np.nonzero(mask)[0]  # already ascending
        deltas = np.diff(np.concatenate(([0], high_idx_sorted)))
        max_delta = int(deltas.max()) if deltas.size > 0 else 0
        idx_dtype = mp.index_dtype_for_max_delta(max_delta)
        idx_deltas = deltas.astype(idx_dtype)

        use_addr = idx_deltas.nbytes < mask_packed.nbytes
        addr_mode = "addr" if use_addr else "mask"

        payload = {
            "n": n, "n_high": int(high_vals.size), "n_low": int(low_vals.size),
            "h_scale": float(h_scale), "h_zmin": float(h_zmin),
            "l_scale": float(l_scale), "l_zmin": float(l_zmin),
            "high_bits": high_bits, "low_bits": low_bits,
            "addr_mode": addr_mode,
            "high_vals": h_q, "low_packed": low_packed,
        }
        if use_addr:
            payload["idx_deltas"] = idx_deltas
        else:
            payload["mask_packed"] = mask_packed
        return payload

    @staticmethod
    def decode(payload: dict) -> np.ndarray:
        n = payload["n"]
        if payload["addr_mode"] == "addr":
            positions = np.cumsum(payload["idx_deltas"].astype(np.int64))
            mask = np.zeros(n, dtype=bool)
            mask[positions] = True
        else:
            mask = np.unpackbits(payload["mask_packed"])[:n].astype(bool)

        high_deq = mp.dequantize_with_params(payload["high_vals"], payload["h_scale"],
                                              payload["h_zmin"], payload["high_bits"])
        low_q = mp.unpack_nibbles(payload["low_packed"], payload["n_low"])
        low_deq = mp.dequantize_with_params(low_q, payload["l_scale"],
                                             payload["l_zmin"], payload["low_bits"])

        dense = np.empty(n, dtype=np.float32)
        dense[mask] = high_deq
        dense[~mask] = low_deq
        return dense


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
HIGH_RATIO_START = 0.30   # round 1: 30% of each tensor's elements at 8-bit
HIGH_RATIO_END = 0.10     # final round: 10% at 8-bit, 90% at 4-bit -- never 0% anywhere
HIGH_BITS = 8
LOW_BITS = 4
STABILITY_LAMBDA = 0.01
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096   # same rationale as SAC: cheap tensors sent dense fp32

print(f"Using device: {DEVICE}")
print(f"Compression strategy: FGMP | enabled={USE_COMPRESSION} | "
      f"high_ratio {HIGH_RATIO_START}->{HIGH_RATIO_END} (cosine) | "
      f"high_bits={HIGH_BITS} low_bits={LOW_BITS} | stability_lambda={STABILITY_LAMBDA}")


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
        high_ratio = fit_config.get("high_ratio", HIGH_RATIO_START)
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
        comm_dense_bytes = 0
        comm_no_compression_bytes = 0
        transform_time_sec = 0.0

        for name, new_val in new_state.items():
            old_val = old_state[name]
            delta = (new_val - old_val).cpu().numpy()
            comm_no_compression_bytes += delta.astype(np.float32).nbytes

            if (not use_compression or name not in fisher_accum
                    or delta.size <= SMALL_TENSOR_FULL_SEND_THRESHOLD):
                out_arrays.append(delta.astype(np.float32))
                meta.append({"encoded": False, "shape": list(delta.shape), "size": int(delta.size)})
                comm_dense_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            fisher_flat = fisher_accum[name].cpu().numpy().reshape(-1)
            delta_flat = delta.reshape(-1).astype(np.float32)
            payload = mp.encode(delta_flat, fisher_flat, high_ratio, HIGH_BITS, LOW_BITS)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(payload["high_vals"])
            out_arrays.append(payload["low_packed"])
            if payload["addr_mode"] == "addr":
                out_arrays.append(payload["idx_deltas"])
            else:
                out_arrays.append(payload["mask_packed"])
            meta.append({
                "encoded": True, "shape": list(delta.shape), "size": payload["n"],
                "n_high": payload["n_high"], "n_low": payload["n_low"],
                "h_scale": payload["h_scale"], "h_zmin": payload["h_zmin"],
                "l_scale": payload["l_scale"], "l_zmin": payload["l_zmin"],
                "high_bits": payload["high_bits"], "low_bits": payload["low_bits"],
                "addr_mode": payload["addr_mode"],
            })

            addr_bytes = (payload["idx_deltas"].nbytes if payload["addr_mode"] == "addr"
                          else payload["mask_packed"].nbytes)
            comm_dense_bytes += payload["high_vals"].nbytes + payload["low_packed"].nbytes + addr_bytes

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "avg_reg_loss": total_reg_loss / len(self.train_loader),
            "compression_meta": json.dumps(meta),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_no_compression_bytes": comm_no_compression_bytes,
            "fisher_time_sec": fisher_time_sec,
            "transform_time_sec": transform_time_sec,
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
class FGMPStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, use_compression=USE_COMPRESSION,
                 high_ratio_start=HIGH_RATIO_START, high_ratio_end=HIGH_RATIO_END,
                 stability_lambda=STABILITY_LAMBDA, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_compression = use_compression
        self.high_ratio_start = high_ratio_start
        self.high_ratio_end = high_ratio_end
        self.stability_lambda = stability_lambda

        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_fisher_time_sec = 0.0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

    def _high_ratio_for_round(self, server_round):
        frac = (server_round - 1) / max(1, NUM_ROUNDS - 1)
        cos = 0.5 * (1 + np.cos(np.pi * frac))
        return float(self.high_ratio_end + (self.high_ratio_start - self.high_ratio_end) * cos)

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        high_ratio = self._high_ratio_for_round(server_round)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_compression"] = self.use_compression
            fit_ins.config["high_ratio"] = high_ratio
            fit_ins.config["stability_lambda"] = self.stability_lambda
        self._current_high_ratio = high_ratio
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0
        reg_losses = []

        round_comm_dense_bytes = 0
        round_comm_no_compression_bytes = 0
        round_fisher_time_sec = []
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("compression_meta", "[]"))
            reg_losses.append(fit_res.metrics.get("avg_reg_loss", 0.0))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_no_compression_bytes += fit_res.metrics.get("comm_no_compression_bytes", 0)
            round_fisher_time_sec.append(fit_res.metrics.get("fisher_time_sec", 0.0))
            round_transform_time_sec.append(fit_res.metrics.get("transform_time_sec", 0.0))

            _recon_start = time.perf_counter()
            cursor = 0
            for k, m in zip(keys, meta):
                shape = tuple(m["shape"])
                if not m["encoded"]:
                    arr = arrays[cursor]; cursor += 1
                    reconstructed = arr.reshape(shape)
                else:
                    high_vals = arrays[cursor]; cursor += 1
                    low_packed = arrays[cursor]; cursor += 1
                    addr_arr = arrays[cursor]; cursor += 1
                    payload = {
                        "n": m["size"], "n_high": m["n_high"], "n_low": m["n_low"],
                        "h_scale": m["h_scale"], "h_zmin": m["h_zmin"],
                        "l_scale": m["l_scale"], "l_zmin": m["l_zmin"],
                        "high_bits": m["high_bits"], "low_bits": m["low_bits"],
                        "addr_mode": m["addr_mode"], "high_vals": high_vals, "low_packed": low_packed,
                    }
                    if m["addr_mode"] == "addr":
                        payload["idx_deltas"] = addr_arr
                    else:
                        payload["mask_packed"] = addr_arr
                    reconstructed = mp.decode(payload).reshape(shape)

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
        avg_reg = float(np.mean(reg_losses)) if reg_losses else 0.0

        self.total_comm_dense_bytes += round_comm_dense_bytes
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
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | high_ratio={self._current_high_ratio:.3f} | "
              f"avg_reg_loss: {avg_reg:.5f}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB)")
        print(f"  [compute] avg client fisher_time: {avg_fisher_time*1000:.2f}ms | "
              f"avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_fgmp.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.print_overhead_summary()

        return aggregated_params, {"accuracy": acc, "comm_dense_bytes": round_comm_dense_bytes}

    def print_overhead_summary(self):
        print("\n========== OVERHEAD SUMMARY (FGMP, cumulative over the run) ==========")
        print(f"Total communication ACTUALLY SENT  : {self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication with NO compression (dense baseline): {self.total_comm_no_compression_bytes/1e6:.2f} MB")
        if self.total_comm_no_compression_bytes:
            print(f"  -> real compression achieved: {self.total_comm_dense_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline")
        print(f"Total client-side fisher accumulation time (avg client, summed over rounds): {self.total_fisher_time_sec:.2f}s")
        print(f"Total client-side transform time (encode, avg client, summed)  : {self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (decode+scatter, summed)   : {self.total_reconstruct_time_sec:.2f}s")
        print(f"Best accuracy achieved: {self.best_acc:.4f}")

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
        plt.title("Final Confusion Matrix (FGMP)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_fgmp.png")
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

    strategy = FGMPStrategy(
        test_loader=test_loader,
        use_compression=USE_COMPRESSION,
        high_ratio_start=HIGH_RATIO_START,
        high_ratio_end=HIGH_RATIO_END,
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