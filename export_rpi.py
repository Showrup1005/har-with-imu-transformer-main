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
        """Stochastic-rounding quantization -- unbiased in expectation, which
        matters more here since we're now quantizing every single element
        (no high-precision safety net for a protected subset)."""
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
        """Generic sub-byte bit-packer -- any width 1..8, not just nibbles."""
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
        """No selection, no dropping: every element goes through the same
        num_bits quantizer. No mask/addr/submask needed -- decoder knows the
        shape already, so 'which positions' is just 'all of them, in order'."""
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

NUM_CLIENTS = 7
LOCAL_EPOCHS = 5
NUM_ROUNDS = 40

USE_COMPRESSION = True
USE_STABILITY_REG = False
NUM_BITS_START = 4.0     # round 1: generous precision, nothing dropped
NUM_BITS_END = 1.5       # final rounds: coarse but still nonzero for every element
STABILITY_LAMBDA = 0.01
SMALL_TENSOR_FULL_SEND_THRESHOLD = 4096   # cheap tensors still sent dense fp32

print(f"Using device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Compression strategy: FGMP (no-drop, uniform quant) | enabled={USE_COMPRESSION} | "
      f"stability_reg={USE_STABILITY_REG} | "
      f"num_bits {NUM_BITS_START}->{NUM_BITS_END} (cosine, rounded per round) | "
      f"keep_ratio=1.0 always | stability_lambda={STABILITY_LAMBDA}")


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
        use_stability_reg = fit_config.get("use_stability_reg", USE_STABILITY_REG)
        num_bits = int(fit_config.get("num_bits", NUM_BITS_START))
        stability_lambda = fit_config.get("stability_lambda", STABILITY_LAMBDA)

        self.model.train()
        total_loss = 0.0
        total_reg_loss = 0.0

        # Fisher is kept ONLY for the stability regularizer now -- it no
        # longer decides who gets dropped or who gets high/low precision,
        # since nobody is dropped and there's no tier to assign.
        fisher_accum = {
            name: torch.zeros_like(p)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        n_grad_steps = 0

        for _ in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()

                self.optimizer.zero_grad()
                output = self.model({"imu": imu})
                ce_loss = self.criterion(output, label)

                reg_loss = torch.zeros((), device=DEVICE)
                if use_stability_reg and stability_lambda > 0 and n_grad_steps > 0:
                    for name, p in self.model.named_parameters():
                        if name in fisher_accum:
                            f_running = (fisher_accum[name] / n_grad_steps).detach()
                            reg_loss = reg_loss + (f_running * (p - old_state[name]) ** 2).sum()

                loss = ce_loss + stability_lambda * reg_loss
                loss.backward()

                if use_stability_reg:
                    for name, p in self.model.named_parameters():
                        if p.grad is not None and name in fisher_accum:
                            fisher_accum[name] += p.grad.detach() ** 2
                n_grad_steps += 1

                self.optimizer.step()
                total_loss += ce_loss.item()
                total_reg_loss += float(reg_loss.detach().item())

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
            "avg_reg_loss": total_reg_loss / len(self.train_loader),
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
    def __init__(self, test_loader, use_compression=USE_COMPRESSION, use_stability_reg=USE_STABILITY_REG,
                 num_bits_start=NUM_BITS_START, num_bits_end=NUM_BITS_END,
                 stability_lambda=STABILITY_LAMBDA, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_compression = use_compression
        self.use_stability_reg = use_stability_reg
        self.num_bits_start = num_bits_start
        self.num_bits_end = num_bits_end
        self.stability_lambda = stability_lambda

        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

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
            fit_ins.config["use_stability_reg"] = self.use_stability_reg
            fit_ins.config["num_bits"] = num_bits
            fit_ins.config["stability_lambda"] = self.stability_lambda
        self._current_num_bits = num_bits
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
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("compression_meta", "[]"))
            reg_losses.append(fit_res.metrics.get("avg_reg_loss", 0.0))

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
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype, device=global_state[k].device)

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_reg = float(np.mean(reg_losses)) if reg_losses else 0.0

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | num_bits={self._current_num_bits} | "
      f"stability_reg={self.use_stability_reg} | avg_reg_loss: {avg_reg:.5f}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB)")
        print(f"  [compute] avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), f"best_model_seed{SEED}.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.print_overhead_summary()

        return aggregated_params, {"accuracy": acc, "comm_dense_bytes": round_comm_dense_bytes}

    def print_overhead_summary(self):
        print(f"\n========== OVERHEAD SUMMARY (FGMP, seed={SEED}, cumulative over the run) ==========")
        print(f"Total communication ACTUALLY SENT  : {self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication with NO compression (dense baseline): {self.total_comm_no_compression_bytes/1e6:.2f} MB")
        if self.total_comm_no_compression_bytes:
            print(f"  -> real compression achieved: {self.total_comm_dense_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline")
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
        plt.title(f"Final Confusion Matrix (FGMP, seed={SEED})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"final_confusion_matrix_seed{SEED}.png")
        plt.close()
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
        use_stability_reg=USE_STABILITY_REG,
        num_bits_start=NUM_BITS_START,
        num_bits_end=NUM_BITS_END,
        stability_lambda=STABILITY_LAMBDA,
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