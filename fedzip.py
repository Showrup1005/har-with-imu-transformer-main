"""
FedZip (Malekijoo et al., 2021 -- arXiv:2102.01593), implemented against
the same FL loop, model, and byte-accounting methodology used in the
SAC script, so its numbers are directly comparable to yours.

Pipeline per tensor (matches the paper's Algorithm 2, encoding method 3
-- "Difference of Address Position", their best-performing variant):

  1. TOP-Z SPARSIFICATION
     Keep only the top-z fraction of a tensor's elements by absolute
     magnitude; zero the rest. Unlike SAC, this is a plain magnitude
     threshold, not Fisher-sensitivity-based -- that's the paper's
     method, not an omission on our part.

  2. K-MEANS QUANTIZATION (k=3)
     Every element of the (sparsified) tensor -- including the zeroed
     ones -- is assigned to the nearest of 3 cluster centroids fit by
     1D k-means. One cluster absorbs the zeroed mass and becomes the
     "majority" cluster; the paper picked k=3 via silhouette index.

  3. ADDRESS-TABLE ENCODING (difference-of-position)
     Nothing is sent for the majority cluster -- it's implicit at
     decode time (filled with that cluster's centroid). For the two
     minority clusters, only their positions are sent, as deltas
     between consecutive positions (usually small, so narrow dtypes
     suffice) plus 1 bit per element for which of the two minority
     centroids applies.

HONESTY NOTE: the paper's headline compression ratios (up to 1085x)
come from true variable-length (Huffman-style) coding of the position
deltas. This implementation uses real, fixed-width numpy dtypes for
the deltas -- the actual-bytes-sent number is genuine and verifiable,
but will be more modest than the paper's number for the same reason
your SAC "ACTUALLY SENT" line is more modest than a full entropy-coded
bitstream would be. The sub-byte floor line below shows how much
further true entropy coding could take it.

No EWC/stability regularizer, no adaptive dense/sparse switching, no
small-tensor bypass -- none of that is part of FedZip. This is meant
as a faithful baseline, not an enhanced one.
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


# ====================== FEDZIP HELPERS ======================
class fz:
    @staticmethod
    def top_z_mask(flat: np.ndarray, z: float) -> np.ndarray:
        n = flat.size
        if z >= 1.0:
            return np.ones(n, dtype=bool)
        k = max(1, int(np.ceil(z * n)))
        if k >= n:
            return np.ones(n, dtype=bool)
        thresh = np.partition(np.abs(flat), n - k)[n - k]  # kth-largest magnitude, i.e. w_z
        return np.abs(flat) >= thresh

    @staticmethod
    def kmeans_1d(x: np.ndarray, k: int = 3, n_iter: int = 15,
                  sample_size: int = 20000, seed: int = 0) -> np.ndarray:
        """1D Lloyd's-algorithm k-means, fit on a subsample for speed
        (the paper doesn't specify an implementation; this keeps a full
        pass over multi-hundred-thousand-element tensors tractable)."""
        rng = np.random.RandomState(seed)
        n = x.size
        if n == 0:
            return np.zeros(k, dtype=np.float64)
        sample = x if n <= sample_size else rng.choice(x, size=sample_size, replace=False)
        qs = np.linspace(0, 100, k + 2)[1:-1]
        centroids = np.unique(np.percentile(sample, qs).astype(np.float64))
        if centroids.size < k:
            pad = np.array([centroids[-1] + (i + 1) * 1e-6 for i in range(k - centroids.size)])
            centroids = np.concatenate([centroids, pad])
        for _ in range(n_iter):
            d = np.abs(sample[:, None] - centroids[None, :])
            labels = np.argmin(d, axis=1)
            for j in range(k):
                pts = sample[labels == j]
                if pts.size > 0:
                    centroids[j] = pts.mean()
        return centroids

    @staticmethod
    def assign_labels(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        d = np.abs(x[:, None] - centroids[None, :])
        return np.argmin(d, axis=1)

    @staticmethod
    def encode(delta_flat: np.ndarray, top_z: float, n_clusters: int = 3) -> dict:
        size = delta_flat.size
        mask = fz.top_z_mask(delta_flat, top_z)
        sparsified = np.where(mask, delta_flat, 0.0).astype(np.float32)

        centroids = fz.kmeans_1d(sparsified, k=n_clusters)
        labels = fz.assign_labels(sparsified, centroids)
        counts = np.bincount(labels, minlength=n_clusters)
        majority_label = int(np.argmax(counts))

        minority_positions = np.nonzero(labels != majority_label)[0]  # ascending
        minority_cluster_ids = labels[minority_positions]
        remaining_labels = [l for l in range(n_clusters) if l != majority_label]
        bit_of_label = {lbl: i for i, lbl in enumerate(remaining_labels)}
        minor_bits = np.array([bit_of_label[int(l)] for l in minority_cluster_ids], dtype=np.uint8)

        deltas = np.diff(np.concatenate(([0], minority_positions)))
        max_delta = int(deltas.max()) if deltas.size > 0 else 0
        if max_delta <= 255:
            delta_dtype = np.uint8
        elif max_delta <= 65535:
            delta_dtype = np.uint16
        else:
            delta_dtype = np.uint32
        deltas = deltas.astype(delta_dtype)
        packed_bits = np.packbits(minor_bits) if minor_bits.size > 0 else np.zeros(0, dtype=np.uint8)

        return {
            "size": size,
            "n_minor": int(minority_positions.size),
            "majority_label": majority_label,
            "remaining_labels": remaining_labels,
            "centroids": centroids.astype(np.float32),
            "deltas": deltas,
            "packed_bits": packed_bits,
        }

    @staticmethod
    def decode(payload: dict) -> np.ndarray:
        size = payload["size"]
        centroids = payload["centroids"]
        fill_value = centroids[payload["majority_label"]]
        dense = np.full(size, fill_value, dtype=np.float32)
        n_minor = payload["n_minor"]
        if n_minor > 0:
            positions = np.cumsum(payload["deltas"].astype(np.int64))
            bits = np.unpackbits(payload["packed_bits"])[:n_minor]
            remaining_labels = np.array(payload["remaining_labels"])
            dense[positions] = centroids[remaining_labels[bits]]
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
FEDZIP_TOP_Z_WEIGHT = 0.1   # keep top 10% of weight-tensor elements by magnitude
FEDZIP_TOP_Z_BIAS = 0.8     # biases sparsified far less aggressively, per the paper's guidance
FEDZIP_N_CLUSTERS = 3       # k=3, selected via silhouette index in the paper

print(f"Using device: {DEVICE}")
print(f"Compression strategy: FedZip | enabled={USE_COMPRESSION} | "
      f"top_z weight={FEDZIP_TOP_Z_WEIGHT} bias={FEDZIP_TOP_Z_BIAS} | k={FEDZIP_N_CLUSTERS}")


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
class FedZipClient(fl.client.NumPyClient):
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
        comm_bitpacked_bytes = 0
        comm_no_compression_bytes = 0
        transform_time_sec = 0.0

        for name, new_val in new_state.items():
            old_val = old_state[name]
            delta = (new_val - old_val).cpu().numpy()
            comm_no_compression_bytes += delta.astype(np.float32).nbytes

            if not use_compression:
                out_arrays.append(delta.astype(np.float32))
                meta.append({"encoded": False, "shape": list(delta.shape), "size": int(delta.size)})
                comm_dense_bytes += delta.astype(np.float32).nbytes
                comm_bitpacked_bytes += delta.astype(np.float32).nbytes
                continue

            _t0 = time.perf_counter()
            top_z = FEDZIP_TOP_Z_BIAS if "bias" in name else FEDZIP_TOP_Z_WEIGHT
            payload = fz.encode(delta.reshape(-1).astype(np.float32), top_z, FEDZIP_N_CLUSTERS)
            transform_time_sec += time.perf_counter() - _t0

            out_arrays.append(payload["deltas"])
            out_arrays.append(payload["packed_bits"])
            out_arrays.append(payload["centroids"])
            meta.append({
                "encoded": True, "shape": list(delta.shape), "size": payload["size"],
                "n_minor": payload["n_minor"], "majority_label": payload["majority_label"],
                "remaining_labels": payload["remaining_labels"],
            })

            comm_dense_bytes += (payload["deltas"].nbytes + payload["packed_bits"].nbytes
                                 + payload["centroids"].nbytes)
            n_minor = payload["n_minor"]
            max_delta = int(payload["deltas"].max()) if n_minor > 0 else 0
            bits_per_delta = max(1, int(np.ceil(np.log2(max_delta + 1)))) if max_delta > 0 else 1
            comm_bitpacked_bytes += n_minor * (bits_per_delta + 1) / 8.0 + payload["centroids"].nbytes

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "compression_meta": json.dumps(meta),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_bitpacked_bytes": comm_bitpacked_bytes,
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
class FedZipStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, use_compression=USE_COMPRESSION, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_compression = use_compression

        self.total_comm_dense_bytes = 0
        self.total_comm_bitpacked_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_compression"] = self.use_compression
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0

        round_comm_dense_bytes = 0
        round_comm_bitpacked_bytes = 0
        round_comm_no_compression_bytes = 0
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            meta = json.loads(fit_res.metrics.get("compression_meta", "[]"))

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_bitpacked_bytes += fit_res.metrics.get("comm_bitpacked_bytes", 0)
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
                    deltas = arrays[cursor]; cursor += 1
                    packed_bits = arrays[cursor]; cursor += 1
                    centroids = arrays[cursor]; cursor += 1
                    payload = {
                        "size": m["size"], "n_minor": m["n_minor"],
                        "majority_label": m["majority_label"],
                        "remaining_labels": m["remaining_labels"],
                        "centroids": centroids, "deltas": deltas, "packed_bits": packed_bits,
                    }
                    reconstructed = fz.decode(payload).reshape(shape)

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
        self.total_comm_bitpacked_bytes += round_comm_bitpacked_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f}")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB) | "
              f"sub-byte bit-packed floor: {round_comm_bitpacked_bytes/1e6:.3f} MB")
        print(f"  [compute] avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_fedzip.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            self.print_overhead_summary()

        return aggregated_params, {"accuracy": acc, "comm_dense_bytes": round_comm_dense_bytes}

    def print_overhead_summary(self):
        print("\n========== OVERHEAD SUMMARY (FedZip, cumulative over the run) ==========")
        print(f"Total communication ACTUALLY SENT  : {self.total_comm_dense_bytes/1e6:.2f} MB")
        print(f"Total communication with NO compression (dense baseline): {self.total_comm_no_compression_bytes/1e6:.2f} MB")
        if self.total_comm_no_compression_bytes:
            print(f"  -> real compression achieved: {self.total_comm_dense_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline "
                  f"({self.total_comm_no_compression_bytes/self.total_comm_dense_bytes:.2f}x)")
            print(f"  -> sub-byte bit-packing floor would be {self.total_comm_bitpacked_bytes/self.total_comm_no_compression_bytes*100:.1f}% "
                  f"of no-compression baseline (what true Huffman/entropy coding of the deltas could reach)")
        print(f"Total client-side transform time (topz+kmeans+encode, avg client, summed): {self.total_transform_time_sec:.2f}s")
        print(f"Total server-side reconstruct time (decode+scatter, summed): {self.total_reconstruct_time_sec:.2f}s")
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
        plt.title("Final Confusion Matrix (FedZip)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_fedzip.png")
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
        return FedZipClient(client_datasets[client_idx]).to_client()

    strategy = FedZipStrategy(test_loader=test_loader, use_compression=USE_COMPRESSION)

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