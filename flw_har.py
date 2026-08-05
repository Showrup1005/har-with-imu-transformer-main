"""
Federated HAR training with Sensitivity-Aware Private Masking (SAPM).

Drop-in replacement for the original fl_train.py. Model, dataset, and
overall FL loop are unchanged; the client no longer sends raw updated
weights -- it sends a Fisher-sparsified, quantized, seed-permuted delta.
The server strategy reverses this before aggregating.
    USE_PRIVACY          -- master on/off switch (False = behaves like
                             the original plain FedAvg script)
    PRIVACY_KEEP_RATIO    -- fraction of each tensor's elements sent
                             per round (Fisher top-k), e.g. 0.4 = 40%
    PRIVACY_QUANT_BITS    -- bits used for stochastic quantization of
                             the transmitted values (8 is a reasonable
                             default; try 4 for a more aggressive test)

OVERHEAD INSTRUMENTATION (NEW):
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
      - transform_time_sec (client): time spent on the post-training
        top-k masking + quantization + permutation pass.
      - reconstruct_time_sec (server): time spent on the matching
        unpermute + dequantize pass across all clients' tensors.

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

# ---- overhead-accounting knobs ----
INDEX_BYTES_PER_ELEMENT = 4  # bytes needed per sparse-encoded index (int32);
                              # used only for the comm_sparse_encoded_bytes
                              # estimate, not for the actual bytes sent.

print(f"Using device: {DEVICE}")
print(f"Privacy strategy: SAPM | enabled={USE_PRIVACY} | keep_ratio={PRIVACY_KEEP_RATIO} | quant_bits={PRIVACY_QUANT_BITS}")

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

        out_arrays = []
        meta = []
        nz_total, elem_total = 0, 0

        # ---- Overhead accounting (see module docstring) ----
        comm_dense_bytes = 0          # what THIS script actually sends
        comm_sparse_encoded_bytes = 0  # what a real sparse encoding would send
        comm_no_privacy_bytes = 0     # plain dense float32 baseline
        transform_time_sec = 0.0      # top-k mask + quantize + permute wall time

        for name, new_val in new_state.items():
            old_val = old_state[name]
            delta = (new_val - old_val).cpu().numpy()
            comm_no_privacy_bytes += delta.astype(np.float32).nbytes

            if not use_privacy or name not in fisher_accum:
                out_arrays.append(delta.astype(np.float32))
                meta.append([1.0, 0.0, False])  # scale, zmin, quantized?
                nz_total += np.count_nonzero(delta)
                elem_total += delta.size
                comm_dense_bytes += delta.astype(np.float32).nbytes
                comm_sparse_encoded_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            fisher_flat = fisher_accum[name].cpu().numpy().reshape(-1)
            delta_flat = delta.reshape(-1).astype(np.float32)

            mask = pu.compute_topk_mask(fisher_flat, keep_ratio)
            sparse_delta = np.where(mask, delta_flat, 0.0).astype(np.float32)

            scale, zmin = pu.compute_quant_params(delta_flat)
            q = pu.quantize_with_params(sparse_delta, scale, zmin, quant_bits)
            permuted = pu.permute_array(q, seed=round_seed * 100003 + deterministic_hash(name) % 97)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(permuted.reshape(delta.shape).astype(np.float32))
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

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
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
                 privacy_keep_ratio=PRIVACY_KEEP_RATIO,
                 privacy_quant_bits=PRIVACY_QUANT_BITS, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_privacy = use_privacy
        self.privacy_keep_ratio = privacy_keep_ratio
        self.privacy_quant_bits = privacy_quant_bits

        # Running totals across the whole run, for the final summary.
        self.total_comm_dense_bytes = 0
        self.total_comm_sparse_encoded_bytes = 0
        self.total_comm_no_privacy_bytes = 0
        self.total_fisher_time_sec = 0.0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        # Same seed broadcast to every client this round: server (and only
        # the server) can invert every client's permutation deterministically.
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_privacy"] = self.use_privacy
            fit_ins.config["privacy_keep_ratio"] = self.privacy_keep_ratio
            fit_ins.config["privacy_quant_bits"] = self.privacy_quant_bits
            fit_ins.config["privacy_seed"] = server_round
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
              f"avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

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
        print(f"Total client-side transform time (mask+quant+permute, avg client, summed)  : "
              f"{self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (unpermute+dequantize, summed over rounds): "
              f"{self.total_reconstruct_time_sec:.2f}s")

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
        plt.title("Final Confusion Matrix (SAPM)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_sapm.png")
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