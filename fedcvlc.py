"""
Fed-CVLC (Su, Zhou, Cui, Lui, Liu -- INFOCOM 2024, arXiv:2402.03770)

Core idea, distinct from every other method in this comparison: instead
of one fixed bit-width for whatever survives sparsification, Fed-CVLC
ranks the kept updates by magnitude and assigns EACH RANK BAND its own
code length -- larger-magnitude updates get more bits, smaller ones
get fewer. The paper shows quantization and Top-k sparsification are
both special cases of this: uniform bit-width for everyone = plain
quantization; bit-width 0 below a cutoff = plain Top-k.

  1. TOP-K SPARSIFICATION (magnitude-based, matching the paper -- no
     Fisher information is used, unlike SAC/FGMP)
  2. VARIABLE-LENGTH GROUPING: the kept elements, ranked descending by
     |value|, are split into Y bands. Band 1 (largest magnitudes,
     fewest elements) gets CODE_LENGTHS[0] bits; band Y (smallest
     magnitudes among the kept, most elements) gets CODE_LENGTHS[-1]
     bits.
  3. UNBIASED QUANTIZATION per band (stochastic rounding, same
     technique used throughout this comparison set).
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


# ====================== GENERIC BIT-PACKING ======================
def pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes (each in [0, 2**bits - 1]) into a byte array,
    `bits` bits per code, MSB-first. Works for any bits in 1..8."""
    if codes.size == 0:
        return np.zeros(0, dtype=np.uint8)
    codes = codes.astype(np.uint32)
    bit_rows = np.zeros((codes.size, bits), dtype=np.uint8)
    for b in range(bits):
        bit_rows[:, b] = (codes >> (bits - 1 - b)) & 1
    return np.packbits(bit_rows.reshape(-1))


def unpack_bits(packed: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Reverse of pack_bits."""
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    flat = np.unpackbits(packed)[: n * bits].reshape(n, bits)
    codes = np.zeros(n, dtype=np.uint32)
    for b in range(bits):
        codes = (codes << 1) | flat[:, b].astype(np.uint32)
    return codes


# ====================== FED-CVLC HELPERS ======================
class cvlc:
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
        """Unbiased stochastic-rounding quantization -- same technique
        used throughout this comparison set, and the property the
        paper's Lemma 1 requires (E[quantized] = original)."""
        if x.size == 0 or num_bits <= 0:
            return np.zeros_like(x)
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        x_scaled = (x - zmin) / step
        floor = np.floor(x_scaled)
        prob = np.clip(x_scaled - floor, 0.0, 1.0)
        rnd = np.random.rand(*x.shape)
        x_q = floor + (rnd < prob)
        return np.clip(x_q, 0, qmax).astype(np.float64)

    @staticmethod
    def dequantize_with_params(x_q: np.ndarray, scale: float, zmin: float, num_bits: int) -> np.ndarray:
        qmax = 2 ** num_bits - 1
        step = scale / qmax if scale != 0 else 1.0
        return x_q.astype(np.float32) * step + zmin

    @staticmethod
    def index_dtype_for_max_delta(max_delta: int):
        if max_delta <= 255:
            return np.uint8
        elif max_delta <= 65535:
            return np.uint16
        else:
            return np.uint32

    @staticmethod
    def encode(delta_flat: np.ndarray, top_k_ratio: float, group_fracs, code_lengths) -> dict:
        n = delta_flat.size
        k = max(1, int(np.ceil(top_k_ratio * n)))
        if k >= n:
            kept_idx = np.arange(n)
        else:
            kept_idx = np.argpartition(np.abs(delta_flat), -k)[-k:]
        k = kept_idx.size

        # Rank kept elements by magnitude, descending -- band 0 gets
        # the largest magnitudes and the most bits.
        mags = np.abs(delta_flat[kept_idx])
        order = np.argsort(-mags)   # descending magnitude
        ranked_idx = kept_idx[order]

        Y = len(code_lengths)
        band_sizes = [max(1, int(round(f * k))) for f in group_fracs]
        # fix rounding so bands sum exactly to k
        band_sizes[-1] = k - sum(band_sizes[:-1])
        band_sizes = [max(0, s) for s in band_sizes]

        bands_idx, bands_vals, bands_scale, bands_zmin = [], [], [], []
        cursor = 0
        for size in band_sizes:
            band_idx = ranked_idx[cursor:cursor + size]
            bands_idx.append(band_idx)
            cursor += size

        for band_idx, bits in zip(bands_idx, code_lengths):
            vals = delta_flat[band_idx]
            scale, zmin = cvlc.compute_quant_params(vals)
            q = cvlc.quantize_with_params(vals, scale, zmin, bits)
            bands_vals.append(q)
            bands_scale.append(scale)
            bands_zmin.append(zmin)

        # Positions of the full kept set within the tensor: bitmask
        # (n bits) vs delta-index (k deltas), whichever is smaller.
        sorted_kept = np.sort(kept_idx)
        mask_full = np.zeros(n, dtype=bool)
        mask_full[sorted_kept] = True
        mask_packed = np.packbits(mask_full)
        deltas = np.diff(np.concatenate(([0], sorted_kept)))
        max_delta = int(deltas.max()) if deltas.size > 0 else 0
        idx_dtype = cvlc.index_dtype_for_max_delta(max_delta)
        idx_deltas = deltas.astype(idx_dtype)
        use_addr = idx_deltas.nbytes < mask_packed.nbytes
        addr_mode = "addr" if use_addr else "mask"

        # Which band each (ascending-order) kept position belongs to --
        # ceil(log2(Y)) bits per kept element.
        band_of_position = {}
        for b, band_idx in enumerate(bands_idx):
            for pos in band_idx:
                band_of_position[int(pos)] = b
        band_id_ascending = np.array([band_of_position[int(p)] for p in sorted_kept], dtype=np.uint32)
        band_id_bits = max(1, int(np.ceil(np.log2(Y))))
        band_id_packed = pack_bits(band_id_ascending, band_id_bits)

        # Pack each band's quantized values (own bit-width), in
        # ASCENDING POSITION order so the decoder can split them back
        # out using band_id_ascending without needing a second index.
        # Rebuild per-band values in ascending-position order:
        vals_by_position = {}
        for band_idx, q, bits in zip(bands_idx, bands_vals, code_lengths):
            for pos, val in zip(band_idx, q):
                vals_by_position[int(pos)] = (val, bits)

        packed_bands = []
        for b, bits in enumerate(code_lengths):
            band_codes_ascending = np.array(
                [vals_by_position[int(p)][0] for p in sorted_kept if band_of_position[int(p)] == b],
                dtype=np.uint32,
            )
            packed_bands.append(pack_bits(band_codes_ascending, bits))

        payload = {
            "n": n, "n_keep": k,
            "band_sizes": band_sizes, "code_lengths": list(code_lengths),
            "band_scales": bands_scale, "band_zmins": bands_zmin,
            "band_id_bits": band_id_bits, "band_id_packed": band_id_packed,
            "packed_bands": packed_bands,
            "addr_mode": addr_mode,
        }
        if use_addr:
            payload["idx_deltas"] = idx_deltas
        else:
            payload["mask_packed"] = mask_packed
        return payload

    @staticmethod
    def decode(payload: dict) -> np.ndarray:
        n = payload["n"]
        n_keep = payload["n_keep"]
        if payload["addr_mode"] == "addr":
            kept_positions = np.cumsum(payload["idx_deltas"].astype(np.int64))
        else:
            kept_positions = np.nonzero(np.unpackbits(payload["mask_packed"])[:n])[0]

        band_id_ascending = unpack_bits(payload["band_id_packed"], payload["band_id_bits"], n_keep)

        code_lengths = payload["code_lengths"]
        band_scales = payload["band_scales"]
        band_zmins = payload["band_zmins"]
        Y = len(code_lengths)

        # Unpack each band's codes, then scatter by matching band_id.
        dense = np.zeros(n, dtype=np.float32)
        cursors = [0] * Y
        band_sizes_actual = [int(np.sum(band_id_ascending == b)) for b in range(Y)]
        band_codes = [
            unpack_bits(payload["packed_bands"][b], code_lengths[b], band_sizes_actual[b])
            for b in range(Y)
        ]
        for i in range(n_keep):
            b = int(band_id_ascending[i])
            code = band_codes[b][cursors[b]]
            cursors[b] += 1
            val = cvlc.dequantize_with_params(np.array([code]), band_scales[b], band_zmins[b], code_lengths[b])[0]
            dense[kept_positions[i]] = val
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
TOP_K_RATIO_START = 0.6     
TOP_K_RATIO_END = 0.15
GROUP_FRACS = [0.10, 0.20, 0.30, 0.40]      # fraction of the kept set in each band, largest-magnitude band first
CODE_LENGTHS = [8, 6, 4, 2]                  # bits for that band -- monotonically fewer bits for smaller magnitudes
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096      

print(f"Using device: {DEVICE}")
print(f"Compression strategy: Fed-CVLC | enabled={USE_COMPRESSION} | "
      f"top_k_ratio {TOP_K_RATIO_START}->{TOP_K_RATIO_END} (cosine) | "
      f"bands={list(zip(GROUP_FRACS, CODE_LENGTHS))}")


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
        top_k_ratio = fit_config.get("top_k_ratio", TOP_K_RATIO_START)

        self.model.train()
        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()
                self.optimizer.zero_grad()
                output = self.model({"imu": imu})
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

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

            if not use_compression or delta.size <= SMALL_TENSOR_FULL_SEND_THRESHOLD:
                out_arrays.append(delta.astype(np.float32))
                meta.append({"encoded": False, "shape": list(delta.shape), "size": int(delta.size)})
                comm_dense_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            delta_flat = delta.reshape(-1).astype(np.float32)
            payload = cvlc.encode(delta_flat, top_k_ratio, GROUP_FRACS, CODE_LENGTHS)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(payload["band_id_packed"])
            for pb in payload["packed_bands"]:
                out_arrays.append(pb)
            if payload["addr_mode"] == "addr":
                out_arrays.append(payload["idx_deltas"])
            else:
                out_arrays.append(payload["mask_packed"])

            meta.append({
                "encoded": True, "shape": list(delta.shape), "size": payload["n"],
                "n_keep": payload["n_keep"], "band_sizes": payload["band_sizes"],
                "code_lengths": payload["code_lengths"],
                "band_scales": payload["band_scales"], "band_zmins": payload["band_zmins"],
                "band_id_bits": payload["band_id_bits"], "addr_mode": payload["addr_mode"],
            })

            addr_bytes = (payload["idx_deltas"].nbytes if payload["addr_mode"] == "addr"
                          else payload["mask_packed"].nbytes)
            band_bytes = sum(pb.nbytes for pb in payload["packed_bands"])
            comm_dense_bytes += payload["band_id_packed"].nbytes + band_bytes + addr_bytes

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "compression_meta": json.dumps(meta),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_no_compression_bytes": comm_no_compression_bytes,
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
class FedCVLCStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, use_compression=USE_COMPRESSION,
                 top_k_start=TOP_K_RATIO_START, top_k_end=TOP_K_RATIO_END, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_compression = use_compression
        self.top_k_start = top_k_start
        self.top_k_end = top_k_end

        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

    def _top_k_for_round(self, server_round):
        frac = (server_round - 1) / max(1, NUM_ROUNDS - 1)
        cos = 0.5 * (1 + np.cos(np.pi * frac))
        return float(self.top_k_end + (self.top_k_start - self.top_k_end) * cos)

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        top_k_ratio = self._top_k_for_round(server_round)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_compression"] = self.use_compression
            fit_ins.config["top_k_ratio"] = top_k_ratio
        self._current_top_k = top_k_ratio
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0

        round_comm_dense_bytes = 0
        round_comm_no_compression_bytes = 0
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("compression_meta", "[]"))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_no_compression_bytes += fit_res.metrics.get("comm_no_compression_bytes", 0)
            round_transform_time_sec.append(fit_res.metrics.get("transform_time_sec", 0.0))

            _recon_start = time.perf_counter()
            cursor = 0
            for k, m in zip(keys, meta):
                shape = tuple(m["shape"])
                if not m["encoded"]:
                    arr = arrays[cursor]; cursor += 1
                    reconstructed = arr.reshape(shape)
                else:
                    band_id_packed = arrays[cursor]; cursor += 1
                    Y = len(m["code_lengths"])
                    packed_bands = []
                    for _ in range(Y):
                        packed_bands.append(arrays[cursor]); cursor += 1
                    addr_arr = arrays[cursor]; cursor += 1
                    payload = {
                        "n": m["size"], "n_keep": m["n_keep"],
                        "code_lengths": m["code_lengths"],
                        "band_scales": m["band_scales"], "band_zmins": m["band_zmins"],
                        "band_id_bits": m["band_id_bits"], "band_id_packed": band_id_packed,
                        "packed_bands": packed_bands, "addr_mode": m["addr_mode"],
                    }
                    if m["addr_mode"] == "addr":
                        payload["idx_deltas"] = addr_arr
                    else:
                        payload["mask_packed"] = addr_arr
                    reconstructed = cvlc.decode(payload).reshape(shape)

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

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | top_k_ratio={self._current_top_k:.3f}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB)")
        print(f"  [compute] avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_fedcvlc.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.print_overhead_summary()

        return aggregated_params, {"accuracy": acc, "comm_dense_bytes": round_comm_dense_bytes}

    def print_overhead_summary(self):
        print("\n========== OVERHEAD SUMMARY (Fed-CVLC, cumulative over the run) ==========")
        print(f"Total communication ACTUALLY SENT  : {self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication with NO compression (dense baseline): {self.total_comm_no_compression_bytes/1e6:.2f} MB")
        if self.total_comm_no_compression_bytes:
            print(f"  -> real compression achieved: {self.total_comm_dense_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline")
        print(f"Total client-side transform time (rank+band+quant+pack, avg client, summed): {self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (unpack+scatter, summed): {self.total_reconstruct_time_sec:.2f}s")
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
        plt.title("Final Confusion Matrix (Fed-CVLC)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_fedcvlc.png")
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

    strategy = FedCVLCStrategy(
        test_loader=test_loader,
        use_compression=USE_COMPRESSION,
        top_k_start=TOP_K_RATIO_START,
        top_k_end=TOP_K_RATIO_END,
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