import argparse
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


# ====================== ARGS ======================
def parse_args():
    parser = argparse.ArgumentParser(description="FL baseline")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for torch/numpy/CUDA and the client data split (default: 42)")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")

    # ---- NEW: what the client transmits ----
    parser.add_argument("--update_mode", type=str, default="delta",
                        choices=["delta", "full", "grad"],
                        help="delta = w_new - w_old (baseline) | "
                             "full  = the new weights themselves | "
                             "grad  = mean minibatch gradient over local training")
    parser.add_argument("--server_lr", type=float, default=1.0,
                        help="Server step size, only used when update_mode=grad "
                             "(w <- w - server_lr * avg_grad)")
    parser.add_argument("--bits_start", type=float, default=4.0)
    parser.add_argument("--bits_end", type=float, default=1.5)

    # ---- NEW: bit-packing ablation ----
    parser.add_argument("--no_bitpack", action="store_true",
                        help="Disable sub-byte bit packing: each quantized value is sent "
                             "as one uint8 (8 bits/element regardless of num_bits). "
                             "Quantization itself is unchanged.")

    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_args()
SEED = ARGS.seed
UPDATE_MODE = ARGS.update_mode
SERVER_LR = ARGS.server_lr
BITPACK = not ARGS.no_bitpack


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
    def encode(flat: np.ndarray, num_bits: int, pack: bool = True) -> dict:
        n = flat.size
        scale, zmin = mp.compute_quant_params(flat)
        q = mp.quantize_with_params(flat, scale, zmin, num_bits).astype(np.uint32)
        if pack:
            packed = mp.pack_bits(q, num_bits)
        else:
            # No packing: one uint8 per element (8 bits on the wire no matter
            # what num_bits is). Values are identical to the packed path.
            assert num_bits <= 8, "unpacked mode stores values in uint8"
            packed = q.astype(np.uint8)
        return {"n": n, "num_bits": num_bits, "scale": float(scale), "zmin": float(zmin),
                "packed": packed, "bitpacked": pack}

    @staticmethod
    def decode(payload: dict) -> np.ndarray:
        n = payload["n"]
        nbits = payload["num_bits"]
        if payload.get("bitpacked", True):
            q = mp.unpack_bits(payload["packed"], n, nbits)
        else:
            q = payload["packed"].astype(np.uint32)
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
NUM_ROUNDS = 40

USE_COMPRESSION = True
NUM_BITS_START = ARGS.bits_start
NUM_BITS_END = ARGS.bits_end
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096   # cheap tensors still sent dense fp32

print(f"Using device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Update mode: {UPDATE_MODE}" + (f" (server_lr={SERVER_LR})" if UPDATE_MODE == "grad" else ""))
print(f"Compression: enabled={USE_COMPRESSION} | num_bits {NUM_BITS_START}->{NUM_BITS_END} (cosine, rounded per round) | bitpack={BITPACK}")


# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    test_dataset = IMUDataset(test_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset

def split_train_data(train_dataset, num_clients=NUM_CLIENTS, seed=SEED):
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
        num_bits = int(fit_config.get("num_bits", NUM_BITS_START))
        update_mode = fit_config.get("update_mode", UPDATE_MODE)
        bitpack = bool(fit_config.get("bitpack", BITPACK))

        # Gradient accumulator (only used in grad mode). Only real parameters
        # have .grad; buffers (if any) fall back to delta in grad mode.
        grad_accum = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        n_steps = 0

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

                if update_mode == "grad":
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_accum[n] += p.grad.detach()
                    n_steps += 1

                self.optimizer.step()
                total_loss += loss.item()

        new_state = self.model.state_dict()

        out_arrays = []
        meta = []
        comm_dense_bytes = 0
        comm_no_compression_bytes = 0
        transform_time_sec = 0.0

        for name, new_val in new_state.items():
            # ---- choose WHAT to transmit for this tensor ----
            if update_mode == "full":
                kind = "full"
                tensor = new_val
            elif update_mode == "grad" and name in grad_accum:
                kind = "grad"
                tensor = grad_accum[name] / max(1, n_steps)
            else:
                kind = "delta"
                tensor = new_val - old_state[name]

            arr = tensor.detach().cpu().numpy()
            comm_no_compression_bytes += arr.astype(np.float32).nbytes

            if (not use_compression or arr.size <= SMALL_TENSOR_FULL_SEND_THRESHOLD):
                out_arrays.append(arr.astype(np.float32))
                meta.append({"encoded": False, "kind": kind, "shape": list(arr.shape), "size": int(arr.size)})
                comm_dense_bytes += arr.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            flat = arr.reshape(-1).astype(np.float32)
            payload = mp.encode(flat, num_bits, pack=bitpack)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(payload["packed"])
            meta.append({
                "encoded": True, "kind": kind, "shape": list(arr.shape), "size": payload["n"],
                "num_bits": payload["num_bits"], "scale": payload["scale"], "zmin": payload["zmin"],
                "bitpacked": bitpack,
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
    def __init__(self, test_loader, use_compression=USE_COMPRESSION,
                 num_bits_start=NUM_BITS_START, num_bits_end=NUM_BITS_END,
                 update_mode=UPDATE_MODE, server_lr=SERVER_LR, bitpack=BITPACK, **kwargs):
        super().__init__(**kwargs)
        self.bitpack = bitpack
        self.run_tag = f"{update_mode}" + ("" if bitpack else "_nopack")
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_compression = use_compression
        self.num_bits_start = num_bits_start
        self.num_bits_end = num_bits_end
        self.update_mode = update_mode
        self.server_lr = server_lr

        # FIX: make clients start round 1 from the SAME weights the server
        # aggregates onto. Without this, Flower asks a random client for its
        # init weights, while global_model has a different random init, so the
        # round-1 delta/grad gets applied to the wrong base.
        self.initial_parameters = ndarrays_to_parameters(
            [v.cpu().numpy() for v in self.global_model.state_dict().values()]
        )

        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

        self.history = {
            "round": [],
            "accuracy": [],
            "num_bits": [],
            "comm_dense_bytes": [],
            "comm_no_compression_bytes": [],
        }

    def _num_bits_for_round(self, server_round):
        frac = (server_round - 1) / max(1, NUM_ROUNDS - 1)
        cos = 0.5 * (1 + np.cos(np.pi * frac))
        bits = self.num_bits_end + (self.num_bits_start - self.num_bits_end) * cos
        return int(max(1, round(bits)))

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        num_bits = self._num_bits_for_round(server_round)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_compression"] = self.use_compression
            fit_ins.config["num_bits"] = num_bits
            fit_ins.config["update_mode"] = self.update_mode
            fit_ins.config["bitpack"] = self.bitpack
        self._current_num_bits = num_bits
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_sum = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        kind_of = {}
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
                kind_of[k] = m.get("kind", "delta")
                if not m["encoded"]:
                    arr = arrays[cursor]; cursor += 1
                    reconstructed = arr.reshape(shape)
                else:
                    packed = arrays[cursor]; cursor += 1
                    payload = {
                        "n": m["size"], "num_bits": m["num_bits"],
                        "scale": m["scale"], "zmin": m["zmin"], "packed": packed,
                        "bitpacked": m.get("bitpacked", True),
                    }
                    reconstructed = mp.decode(payload).reshape(shape)

                weighted_sum[k] += reconstructed.astype(np.float64) * num_examples
            round_reconstruct_time_sec += time.perf_counter() - _recon_start
            total_examples += num_examples

        new_state = {}
        for k in keys:
            avg = weighted_sum[k] / max(1, total_examples)
            base = global_state[k].cpu().numpy().astype(np.float64)
            kind = kind_of.get(k, "delta")

            if kind == "full":
                new_val = avg                              # FedAvg on weights
            elif kind == "grad":
                new_val = base - self.server_lr * avg      # server-side SGD step
            else:
                new_val = base + avg                       # FedAvg on deltas

            dt = global_state[k].dtype
            if dt.is_floating_point:
                t = torch.tensor(new_val, dtype=dt)
            else:                                          # e.g. int64 buffers
                t = torch.tensor(np.rint(new_val), dtype=dt)
            new_state[k] = t.to(global_state[k].device)

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc, precision_w, recall_w, f1_w, precision_m, recall_m, f1_m, auc_macro = self.evaluate_global(final=False)

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        self.history["round"].append(server_round)
        self.history["accuracy"].append(acc)
        self.history["num_bits"].append(self._current_num_bits)
        self.history["comm_dense_bytes"].append(round_comm_dense_bytes)
        self.history["comm_no_compression_bytes"].append(round_comm_no_compression_bytes)

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} [{self.run_tag}] - Accuracy: {acc:.4f} | num_bits={self._current_num_bits}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB)")
        print(f"  [compute] avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), f"best_model_seed{SEED}_{self.run_tag}.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.save_run_data(path=f"fl_run_history_{self.run_tag}_seed{SEED}.npz")

        return aggregated_params, {"accuracy": acc, "comm_dense_bytes": round_comm_dense_bytes}

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
            "best_acc": self.best_acc,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "use_compression": self.use_compression,
            "num_bits_start": self.num_bits_start,
            "num_bits_end": self.num_bits_end,
            "update_mode": self.update_mode,
            "server_lr": self.server_lr,
            "bitpack": self.bitpack,
        })

        np.savez(path, **save_kwargs)
        print(f"Saved run data to {path}")

    def evaluate_global(self, final=False):
        self.global_model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in self.test_loader:
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
        precision_w = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall_w = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        precision_m = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall_m = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        try:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_auc_score
            classes = np.unique(all_labels)
            y_onehot = label_binarize(all_labels, classes=classes)
            auc_macro = roc_auc_score(y_onehot, all_probs, multi_class="ovr", average="macro")
        except Exception:
            auc_macro = float("nan")

        if not final:
            return accuracy, precision_w, recall_w, f1_w, precision_m, recall_m, f1_m, auc_macro

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision (weighted): {precision_w:.4f} | (macro): {precision_m:.4f}")
        print(f"Recall    (weighted): {recall_w:.4f} | (macro): {recall_m:.4f}")
        print(f"F1        (weighted): {f1_w:.4f} | (macro): {f1_m:.4f}")
        print(f"ROC-AUC (macro, OvR): {auc_macro:.4f}")
        print("\nClassification Report")
        print(classification_report(all_labels, all_preds, zero_division=0))

        self._final_all_labels = all_labels
        self._final_all_preds = all_preds
        self._final_all_probs = all_probs

        return accuracy


# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset, NUM_CLIENTS, seed=SEED)
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
        test_loader=test_loader,
        use_compression=USE_COMPRESSION,
        num_bits_start=NUM_BITS_START,
        num_bits_end=NUM_BITS_END,
        update_mode=UPDATE_MODE,
        server_lr=SERVER_LR,
        bitpack=BITPACK,
    )

    print(f"Starting FL | seed={SEED} | mode={UPDATE_MODE} | bitpack={BITPACK} | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
    )

if __name__ == "__main__":
    main(ARGS.train_csv, ARGS.test_csv)