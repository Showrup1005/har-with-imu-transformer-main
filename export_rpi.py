import argparse
import torch
import numpy as np
import json
import time
import gc
import warnings
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset


# ====================== ARGS ======================
def parse_args():
    parser = argparse.ArgumentParser(description="FL baseline (in-process simulation, no Ray/Flower)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")

    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=80)
    parser.add_argument("--local_epochs", type=int, default=5)

    # compression
    parser.add_argument("--no_compression", action="store_true", help="Send dense fp32 deltas")
    parser.add_argument("--bits_start", type=float, default=4.0)
    parser.add_argument("--bits_end", type=float, default=1.5)

    # validation split
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fraction of train windows held out (server-side) for validation")
    parser.add_argument("--val_block", type=int, default=50,
                        help="Validation windows are held out in contiguous blocks of this size")

    # early stopping (driven by VALIDATION accuracy)
    parser.add_argument("--patience", type=int, default=8, help="0 disables early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--min_rounds", type=int, default=15)

    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_args()
SEED = ARGS.seed
NUM_CLIENTS = ARGS.num_clients
NUM_ROUNDS = ARGS.num_rounds
LOCAL_EPOCHS = ARGS.local_epochs
USE_COMPRESSION = not ARGS.no_compression
NUM_BITS_START = ARGS.bits_start
NUM_BITS_END = ARGS.bits_end
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096


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
        q = mp.unpack_bits(payload["packed"], payload["n"], payload["num_bits"])
        return mp.dequantize_with_params(q, payload["scale"], payload["zmin"], payload["num_bits"])


# ====================== CONFIG / SEEDS ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cpu" if ARGS.cpu or not torch.cuda.is_available() else "cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {DEVICE}")
print(f"Seed: {SEED} | clients={NUM_CLIENTS} | rounds={NUM_ROUNDS} | local_epochs={LOCAL_EPOCHS}")
print(f"Compression: enabled={USE_COMPRESSION} | num_bits {NUM_BITS_START}->{NUM_BITS_END} (cosine, rounded per round)")
print(f"Early stopping on VAL acc: patience={ARGS.patience} min_delta={ARGS.min_delta} "
      f"min_rounds={ARGS.min_rounds} | val_ratio={ARGS.val_ratio}")


# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    test_dataset = IMUDataset(test_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset


def make_val_split(train_dataset, val_ratio, block_size, seed):
    """Server-side validation split in contiguous blocks, with train windows that
    share raw samples with a validation window purged (prevents overlap leakage)."""
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

    ws, sh = config["window_size"], config["window_shift"]
    sh = ws if sh is None else sh
    overlap = int(np.ceil(ws / sh)) - 1

    near_val = val_mask.copy()
    if overlap > 0:
        near_val = np.convolve(val_mask.astype(float), np.ones(2 * overlap + 1), mode="same") > 0

    train_idx = np.where(~near_val)[0]
    val_idx = np.where(val_mask)[0]
    print(f"Validation split: {len(train_idx)} train / {len(val_idx)} val windows "
          f"({n - len(train_idx) - len(val_idx)} purged for overlap, overlap={overlap})")
    return train_idx, val_idx


def split_train_data(train_dataset, num_clients, seed):
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
        print(f"Client {i} -> {len(subset)} samples | Label distribution: "
              f"{dict(zip(unique.tolist(), counts.tolist()))}")
    print("=" * 60)
    return client_datasets


# ====================== CLIENT ======================
class IMUClient:
    """A fresh client (model + optimizer) is built each round, matching how
    Flower's simulation behaved (client_fn was called every round)."""

    def __init__(self, train_subset):
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"],
                                          weight_decay=config.get("weight_decay", 1e-4))
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_parameters(self, params):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, use_compression, num_bits):
        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}

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

        out_arrays, meta = [], []
        comm_dense_bytes = 0
        comm_no_compression_bytes = 0
        transform_time_sec = 0.0

        for name, new_val in new_state.items():
            delta = (new_val - old_state[name]).cpu().numpy()
            comm_no_compression_bytes += delta.astype(np.float32).nbytes

            if (not use_compression) or delta.size <= SMALL_TENSOR_FULL_SEND_THRESHOLD:
                out_arrays.append(delta.astype(np.float32))
                meta.append({"encoded": False, "shape": list(delta.shape), "size": int(delta.size)})
                comm_dense_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            payload = mp.encode(delta.reshape(-1).astype(np.float32), num_bits)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(payload["packed"])
            meta.append({
                "encoded": True, "shape": list(delta.shape), "size": payload["n"],
                "num_bits": payload["num_bits"], "scale": payload["scale"], "zmin": payload["zmin"],
            })
            comm_dense_bytes += payload["packed"].nbytes

        metrics = {
            "train_loss": total_loss / max(1, len(self.train_loader)),
            "compression_meta": meta,
            "comm_dense_bytes": comm_dense_bytes,
            "comm_no_compression_bytes": comm_no_compression_bytes,
            "transform_time_sec": transform_time_sec,
        }
        return out_arrays, len(self.train_loader.dataset), metrics


# ====================== SERVER ======================
class Server:
    def __init__(self, val_loader, test_loader, use_compression=USE_COMPRESSION,
                 num_bits_start=NUM_BITS_START, num_bits_end=NUM_BITS_END,
                 patience=8, min_delta=1e-3, min_rounds=15):
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.use_compression = use_compression
        self.num_bits_start = num_bits_start
        self.num_bits_end = num_bits_end

        # best-model tracking (validation)
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
        self.finished = False

        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

        self.history = {
            "round": [], "val_accuracy": [], "test_accuracy": [],   # test logged only, never used for selection
            "num_bits": [], "comm_dense_bytes": [], "comm_no_compression_bytes": [],
        }

    def get_parameters(self):
        return [v.detach().cpu().numpy() for v in self.global_model.state_dict().values()]

    def num_bits_for_round(self, server_round):
        frac = (server_round - 1) / max(1, NUM_ROUNDS - 1)
        cos = 0.5 * (1 + np.cos(np.pi * frac))
        bits = self.num_bits_end + (self.num_bits_start - self.num_bits_end) * cos
        return int(max(1, round(bits)))

    def aggregate_fit(self, server_round, num_bits, results):
        """results: list of (arrays, num_examples, metrics). Returns new global params."""
        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0

        round_comm_dense_bytes = 0
        round_comm_no_compression_bytes = 0
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for arrays, num_examples, metrics in results:
            meta = metrics["compression_meta"]
            round_comm_dense_bytes += metrics["comm_dense_bytes"]
            round_comm_no_compression_bytes += metrics["comm_no_compression_bytes"]
            round_transform_time_sec.append(metrics["transform_time_sec"])

            _t0 = time.perf_counter()
            for cursor, (k, m) in enumerate(zip(keys, meta)):
                shape = tuple(m["shape"])
                if not m["encoded"]:
                    reconstructed = arrays[cursor].reshape(shape)
                else:
                    payload = {"n": m["size"], "num_bits": m["num_bits"],
                               "scale": m["scale"], "zmin": m["zmin"], "packed": arrays[cursor]}
                    reconstructed = mp.decode(payload).reshape(shape)
                weighted_deltas[k] += reconstructed.astype(np.float64) * num_examples
            round_reconstruct_time_sec += time.perf_counter() - _t0
            total_examples += num_examples

        new_state = {}
        for k in keys:
            avg_delta = weighted_deltas[k] / max(1, total_examples)
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype,
                                                          device=global_state[k].device)
        self.global_model.load_state_dict(new_state)

        # ---- evaluation: val drives decisions, test only logged ----
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
        self.history["num_bits"].append(num_bits)
        self.history["comm_dense_bytes"].append(round_comm_dense_bytes)
        self.history["comm_no_compression_bytes"].append(round_comm_no_compression_bytes)

        ratio = (round_comm_dense_bytes / round_comm_no_compression_bytes
                 if round_comm_no_compression_bytes else 1.0)
        print(f"Round {server_round}/{NUM_ROUNDS} - Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | num_bits={num_bits}")
        print(f"  [comm] SENT: {round_comm_dense_bytes/1e6:.3f} MB ({ratio*100:.1f}% of "
              f"{round_comm_no_compression_bytes/1e6:.3f} MB baseline)")
        print(f"  [compute] avg client transform: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct: {round_reconstruct_time_sec*1000:.2f}ms")

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

        should_stop = (self.patience > 0
                       and server_round >= self.min_rounds
                       and self.rounds_no_improve >= self.patience
                       and server_round < NUM_ROUNDS)

        if should_stop or server_round == NUM_ROUNDS:
            if should_stop:
                self.stopped = True
                self.stopped_round = server_round
                print(f"\n*** Early stopping at round {server_round}: no val improvement "
                      f">{self.min_delta} for {self.patience} rounds "
                      f"(best val acc {self.best_val_acc:.4f} @ round {self.best_round}) ***")
            self.finish()

        return self.get_parameters()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.best_state is not None:
            self.global_model.load_state_dict(self.best_state)
        print(f"\n========== FINAL EVALUATION (best-val model, round {self.best_round}) ==========")
        if self.val_loader is not None:
            print("--- Validation ---")
            self.evaluate_global(self.val_loader, final=True, store=False)
        print("--- Test (reported once, on the restored best model) ---")
        self.evaluate_global(self.test_loader, final=True, store=True)
        self.save_run_data()

    def save_run_data(self, path="fl_run_history.npz"):
        save_kwargs = {k: np.array(v) for k, v in self.history.items()}
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
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
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

    # carve validation out of train (server-side), then split the rest across clients
    train_idx, val_idx = make_val_split(train_dataset, ARGS.val_ratio, ARGS.val_block, SEED)
    val_loader = (DataLoader(Subset(train_dataset, val_idx), batch_size=config["batch_size"], shuffle=False)
                  if len(val_idx) > 0 else None)
    if val_loader is None:
        print("WARNING: no validation set (val_ratio=0) -- selection falls back to the TEST set (leaky).")
    client_datasets = split_train_data(Subset(train_dataset, train_idx), NUM_CLIENTS, seed=SEED)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    server = Server(val_loader, test_loader,
                    use_compression=USE_COMPRESSION,
                    num_bits_start=NUM_BITS_START, num_bits_end=NUM_BITS_END,
                    patience=ARGS.patience, min_delta=ARGS.min_delta, min_rounds=ARGS.min_rounds)

    # all clients start from the server's initial weights
    global_params = server.get_parameters()

    print(f"\nStarting FL (in-process) | seed={SEED} | {NUM_CLIENTS} clients | {NUM_ROUNDS} rounds\n")
    for server_round in range(1, NUM_ROUNDS + 1):
        num_bits = server.num_bits_for_round(server_round)
        results = []
        for cid in range(NUM_CLIENTS):
            client = IMUClient(client_datasets[cid])
            arrays, n, metrics = client.fit(global_params, USE_COMPRESSION, num_bits)
            print(f"  client {cid}: train_loss={metrics['train_loss']:.4f}")
            results.append((arrays, n, metrics))
            del client
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        global_params = server.aggregate_fit(server_round, num_bits, results)
        if server.stopped:
            break

    server.finish()   # no-op if already finished


if __name__ == "__main__":
    main(ARGS.train_csv, ARGS.test_csv)