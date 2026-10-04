import argparse
import flwr as fl
import torch
import numpy as np
import json
import time
import warnings
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# ====================== ARGS ======================
def parse_args():
    parser = argparse.ArgumentParser(description="FL baseline")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for torch/numpy/CUDA and the data splits (default: 42)")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")

    # validation split
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fraction of train windows held out (server-side) for validation")
    parser.add_argument("--val_block", type=int, default=50,
                        help="Validation windows are held out in contiguous blocks of this many windows")

    # early stopping (driven by VALIDATION accuracy)
    parser.add_argument("--patience", type=int, default=8,
                        help="Rounds without val improvement before stopping (0 disables)")
    parser.add_argument("--min_delta", type=float, default=1e-3,
                        help="Minimum val-accuracy gain that counts as an improvement")
    parser.add_argument("--min_rounds", type=int, default=15,
                        help="Never stop before this round")
    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_args()
SEED = ARGS.seed


# ====================== UNIFORM-PRECISION QUANT HELPERS ======================
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
        """Stochastic-rounding quantization -- unbiased in expectation."""
        if x.size == 0:
            return x.astype(np.float32)
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
    def pack_bits(values: np.ndarray, nbits: int) -> np.ndarray:
        """Generic sub-byte bit-packer -- any width 1..8."""
        v = values.astype(np.uint32)
        bit_planes = ((v[:, None] >> np.arange(nbits - 1, -1, -1)) & 1).astype(np.uint8)
        return np.packbits(bit_planes.reshape(-1))

    @staticmethod
    def unpack_bits(packed: np.ndarray, n: int, nbits: int) -> np.ndarray:
        total_bits = n * nbits
        bits = np.unpackbits(packed)[:total_bits].reshape(n, nbits)
        weights = (1 << np.arange(nbits - 1, -1, -1)).astype(np.uint32)
        return (bits * weights).sum(axis=1).astype(np.uint32)

    @staticmethod
    def encode(delta_flat: np.ndarray, num_bits: int) -> dict:
        n = delta_flat.size
        scale, zmin = mp.compute_quant_params(delta_flat)
        q = mp.quantize_with_params(delta_flat, scale, zmin, num_bits).astype(np.uint32)
        packed = mp.pack_bits(q, num_bits)
        return {"n": n, "num_bits": num_bits, "scale": float(scale), "zmin": float(zmin), "packed": packed}

    @staticmethod
    def decode(payload: dict) -> np.ndarray:
        n = payload["n"]
        nbits = payload["num_bits"]
        q = mp.unpack_bits(payload["packed"], n, nbits)
        return mp.dequantize_with_params(q, payload["scale"], payload["zmin"], nbits)


# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_CLIENTS = 3
LOCAL_EPOCHS = 5
NUM_ROUNDS = 80

USE_COMPRESSION = True
NUM_BITS_START = 4.0
NUM_BITS_END = 1.5
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096

print(f"Using device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Compression strategy: Fed-CAUQ (no-drop, uniform quant) | enabled={USE_COMPRESSION} | "
      f"num_bits {NUM_BITS_START}->{NUM_BITS_END} (cosine, rounded per round) | "
      f"keep_ratio=1.0 always")
print(f"Early stopping on VAL acc: patience={ARGS.patience} min_delta={ARGS.min_delta} "
      f"min_rounds={ARGS.min_rounds} | val_ratio={ARGS.val_ratio}")


# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    test_dataset = IMUDataset(test_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset


def make_val_split(train_dataset, val_ratio, block_size, seed):
    """Server-side validation split.

    Windows are held out in contiguous BLOCKS (not individually), and train
    windows that share raw samples with any validation window are purged.
    Otherwise, when window_shift < window_size, neighbouring windows overlap
    and validation would leak into training.
    Returns (train_indices, val_indices) as sorted numpy arrays.
    """
    n = len(train_dataset)
    if val_ratio <= 0:
        return np.arange(n), np.array([], dtype=int)

    rng = np.random.RandomState(seed)
    block_size = max(1, block_size)
    n_blocks = int(np.ceil(n / block_size))
    n_val_blocks = max(1, int(round(n_blocks * val_ratio)))
    val_blocks = rng.choice(n_blocks, size=n_val_blocks, replace=False)

    block_id = np.arange(n) // block_size
    val_mask = np.isin(block_id, val_blocks)

    # number of neighbouring windows on each side that share raw samples
    ws, sh = config["window_size"], config["window_shift"]
    sh = ws if sh is None else sh
    overlap = int(np.ceil(ws / sh)) - 1

    near_val = val_mask.copy()
    if overlap > 0:
        kernel = np.ones(2 * overlap + 1)
        near_val = np.convolve(val_mask.astype(float), kernel, mode="same") > 0

    train_idx = np.where(~near_val)[0]
    val_idx = np.where(val_mask)[0]
    print(f"Validation split: {len(train_idx)} train / {len(val_idx)} val windows "
          f"({n - len(train_idx) - len(val_idx)} purged for overlap, overlap={overlap})")
    return train_idx, val_idx


def split_train_data(train_dataset, num_clients=NUM_CLIENTS, seed=SEED):
    n = len(train_dataset)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"],
                                          weight_decay=config.get("weight_decay", 1e-4))
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
        num_bits = int(fit_config.get("num_bits", NUM_BITS_START))

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

            if (not use_compression or delta.size <= SMALL_TENSOR_FULL_SEND_THRESHOLD):
                out_arrays.append(delta.astype(np.float32))
                meta.append({"encoded": False, "shape": list(delta.shape), "size": int(delta.size)})
                comm_dense_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            delta_flat = delta.reshape(-1).astype(np.float32)
            payload = mp.encode(delta_flat, num_bits)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(payload["packed"])
            meta.append({
                "encoded": True, "shape": list(delta.shape), "size": payload["n"],
                "num_bits": payload["num_bits"], "scale": payload["scale"], "zmin": payload["zmin"],
            })
            comm_dense_bytes += payload["packed"].nbytes

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
class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, val_loader, test_loader, use_compression=USE_COMPRESSION,
                 num_bits_start=NUM_BITS_START, num_bits_end=NUM_BITS_END,
                 patience=8, min_delta=1e-3, min_rounds=15, **kwargs):
        super().__init__(**kwargs)
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.use_compression = use_compression
        self.num_bits_start = num_bits_start
        self.num_bits_end = num_bits_end

        # best-model tracking (by VALIDATION accuracy)
        self.best_val_acc = -1.0
        self.best_round = 0
        self.best_state = None

        # early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.es_best = -1.0
        self.rounds_no_improve = 0
        self.stopped = False
        self.stopped_round = None

        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

        self.history = {
            "round": [],
            "val_accuracy": [],
            "test_accuracy": [],   # logged for curves only; never used for selection
            "num_bits": [],
            "comm_dense_bytes": [],
            "comm_no_compression_bytes": [],
        }

    def initialize_parameters(self, client_manager):
        # Start every client from the SAME weights the server delta-accumulates onto.
        # (Otherwise Flower pulls the initial weights from a random client, whose
        # init differs from self.global_model.)
        return ndarrays_to_parameters([v.cpu().numpy() for v in self.global_model.state_dict().values()])

    def _num_bits_for_round(self, server_round):
        frac = (server_round - 1) / max(1, NUM_ROUNDS - 1)
        cos = 0.5 * (1 + np.cos(np.pi * frac))
        bits = self.num_bits_end + (self.num_bits_start - self.num_bits_end) * cos
        return int(max(1, round(bits)))

    def configure_fit(self, server_round, parameters, client_manager):
        if self.stopped:
            return []
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        num_bits = self._num_bits_for_round(server_round)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_compression"] = self.use_compression
            fit_ins.config["num_bits"] = num_bits
        self._current_num_bits = num_bits
        return fit_ins_list

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.stopped:
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

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
                    packed = arrays[cursor]; cursor += 1
                    payload = {
                        "n": m["size"], "num_bits": m["num_bits"],
                        "scale": m["scale"], "zmin": m["zmin"], "packed": packed,
                    }
                    reconstructed = mp.decode(payload).reshape(shape)

                weighted_deltas[k] += reconstructed.astype(np.float64) * num_examples
            round_reconstruct_time_sec += time.perf_counter() - _recon_start
            total_examples += num_examples

        new_state = {}
        for k in keys:
            avg_delta = weighted_deltas[k] / max(1, total_examples)
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype,
                                                          device=global_state[k].device)

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        # ---- evaluation: val drives decisions, test is only logged ----
        has_val = self.val_loader is not None
        val_acc = self.evaluate_global(self.val_loader)["accuracy"] if has_val else float("nan")
        test_acc = self.evaluate_global(self.test_loader)["accuracy"]
        select_acc = val_acc if has_val else test_acc   # fallback if val_ratio=0 (leaky!)

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        self.history["round"].append(server_round)
        self.history["val_accuracy"].append(val_acc)
        self.history["test_accuracy"].append(test_acc)
        self.history["num_bits"].append(self._current_num_bits)
        self.history["comm_dense_bytes"].append(round_comm_dense_bytes)
        self.history["comm_no_compression_bytes"].append(round_comm_no_compression_bytes)

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} "
              f"| num_bits={self._current_num_bits}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: "
              f"{round_comm_no_compression_bytes/1e6:.3f} MB)")
        print(f"  [compute] avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        # ---- best-model tracking (validation) ----
        if select_acc > self.best_val_acc:
            self.best_val_acc = select_acc
            self.best_round = server_round
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
            torch.save(self.global_model.state_dict(), f"best_model_seed{SEED}.pth")

        # ---- early stopping (validation) ----
        if select_acc > self.es_best + self.min_delta:
            self.es_best = select_acc
            self.rounds_no_improve = 0
        else:
            self.rounds_no_improve += 1

        should_stop = (
            self.patience > 0
            and server_round >= self.min_rounds
            and self.rounds_no_improve >= self.patience
            and server_round < NUM_ROUNDS
        )

        if should_stop or server_round == NUM_ROUNDS:
            if should_stop:
                self.stopped = True
                self.stopped_round = server_round
                print(f"\n*** Early stopping at round {server_round}: no val improvement "
                      f">{self.min_delta} for {self.patience} rounds "
                      f"(best val acc {self.best_val_acc:.4f} @ round {self.best_round}) ***")
            if self.best_state is not None:
                self.global_model.load_state_dict(self.best_state)
                aggregated_params = ndarrays_to_parameters(
                    [v.cpu().numpy() for v in self.global_model.state_dict().values()])
            print(f"\n========== FINAL EVALUATION (best-val model, round {self.best_round}) ==========")
            print("--- Validation ---")
            if has_val:
                self.evaluate_global(self.val_loader, final=True, store=False)
            print("--- Test (reported once, on the restored best model) ---")
            self.evaluate_global(self.test_loader, final=True, store=True)
            self.save_run_data()

        return aggregated_params, {"val_accuracy": val_acc, "test_accuracy": test_acc,
                                   "comm_dense_bytes": round_comm_dense_bytes}

    def save_run_data(self, path="fl_run_history.npz"):
        h = self.history
        save_kwargs = {k: np.array(v) for k, v in h.items()}

        save_kwargs.update({
            "final_labels": getattr(self, "_final_all_labels", np.array([])),
            "final_preds": getattr(self, "_final_all_preds", np.array([])),
            "final_probs": getattr(self, "_final_all_probs", np.array([])),
            "total_comm_dense_bytes": self.total_comm_dense_bytes,
            "total_comm_no_compression_bytes": self.total_comm_no_compression_bytes,
            "total_transform_time_sec": self.total_transform_time_sec,
            "total_reconstruct_time_sec": self.total_reconstruct_time_sec,
            "best_val_acc": self.best_val_acc,
            "best_round": self.best_round,
            "final_test_acc": getattr(self, "_final_test_acc", float("nan")),
            "num_rounds": NUM_ROUNDS,
            "stopped_round": self.stopped_round if self.stopped_round is not None else NUM_ROUNDS,
            "early_stopped": self.stopped,
            "num_clients": NUM_CLIENTS,
            "use_compression": self.use_compression,
            "num_bits_start": self.num_bits_start,
            "num_bits_end": self.num_bits_end,
        })

        np.savez(path, **save_kwargs)
        print(f"Saved run data to {path}")

    def evaluate_global(self, loader, final=False, store=False):
        self.global_model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()
                outputs = self.global_model({"imu": imu})
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        if not final:
            return {"accuracy": accuracy}

        precision_w = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall_w = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        precision_m = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall_m = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        try:
            classes = np.unique(all_labels)
            y_onehot = label_binarize(all_labels, classes=classes)
            auc_macro = roc_auc_score(y_onehot, all_probs[:, classes], multi_class="ovr", average="macro")
        except Exception:
            auc_macro = float("nan")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision (weighted): {precision_w:.4f} | (macro): {precision_m:.4f}")
        print(f"Recall    (weighted): {recall_w:.4f} | (macro): {recall_m:.4f}")
        print(f"F1        (weighted): {f1_w:.4f} | (macro): {f1_m:.4f}")
        print(f"ROC-AUC (macro, OvR): {auc_macro:.4f}")
        print("\nClassification Report")
        print(classification_report(all_labels, all_preds, zero_division=0))

        if store:
            self._final_all_labels = all_labels
            self._final_all_preds = all_preds
            self._final_all_probs = all_probs
            self._final_test_acc = accuracy

        return {"accuracy": accuracy}


# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)

    # 1) carve the validation set out of the training data (server-side)
    train_idx, val_idx = make_val_split(train_dataset, ARGS.val_ratio, ARGS.val_block, SEED)
    train_part = Subset(train_dataset, train_idx)
    val_loader = (DataLoader(Subset(train_dataset, val_idx), batch_size=config["batch_size"], shuffle=False)
                  if len(val_idx) > 0 else None)
    if val_loader is None:
        print("WARNING: no validation set (val_ratio=0) -- early stopping falls back to the TEST set (leaky).")

    # 2) split ONLY the remaining train windows across clients
    client_datasets = split_train_data(train_part, NUM_CLIENTS, seed=SEED)
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

    strategy = Strategy(
        val_loader=val_loader,
        test_loader=test_loader,
        use_compression=USE_COMPRESSION,
        num_bits_start=NUM_BITS_START,
        num_bits_end=NUM_BITS_END,
        patience=ARGS.patience,
        min_delta=ARGS.min_delta,
        min_rounds=ARGS.min_rounds,
    )

    print(f"Starting FL | seed={SEED} | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
    )


if __name__ == "__main__":
    main(ARGS.train_csv, ARGS.test_csv)