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

print(f"Using device: {DEVICE}")
print(f"Strategy: int8 (no stability regularizer)")

# ============================================================
# communication cost measurement utilities
# ============================================================
def compute_ndarrays_size(ndarrays):
    """Total size in bytes of a list of numpy arrays, as they'll actually
    be sent over the wire (each array's own dtype, not assumed fp32)."""
    return sum(arr.nbytes for arr in ndarrays)


def print_model_size_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / (1024 ** 2)
    n_tensors = sum(1 for _ in model.parameters())
    scale_overhead_bytes = n_tensors * 4  # one float32 scale per tensor

    print("\n" + "=" * 60)
    print("MODEL SIZE SUMMARY")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (fp32):    {size_bytes:,} bytes = {size_mb:.2f} MB")
    print(f"Model size (int8):    {size_bytes // 4 + scale_overhead_bytes:,} bytes "
          f"= {(size_bytes / 4 + scale_overhead_bytes) / (1024**2):.2f} MB  "
          f"(expected with quantization, incl. {scale_overhead_bytes} bytes scale overhead)")
    print("=" * 60 + "\n")
    return size_bytes


class CommunicationTracker:
    """Accumulates real transmitted bytes across the whole FL run.

    Upload only: client -> server (parameters returned from fit).
    Download (server -> client) is intentionally not tracked.
    """
    def __init__(self):
        self.round_log = []
        self.total_upload_bytes = 0

    def log_round(self, server_round, upload_bytes):
        self.total_upload_bytes += upload_bytes
        self.round_log.append({
            "round": server_round,
            "upload_bytes": upload_bytes,
        })
        print(f"  [Comm] Round {server_round}: upload={upload_bytes/1024**2:.2f} MB")

    def summary(self):
        print("\n" + "=" * 60)
        print("COMMUNICATION COST SUMMARY (UPLOAD ONLY)")
        print("=" * 60)
        print(f"Total upload (client->server): {self.total_upload_bytes/1024**2:.2f} MB "
              f"({self.total_upload_bytes/1024**3:.4f} GB)")
        if self.round_log:
            avg_round = self.total_upload_bytes / len(self.round_log)
            print(f"Average upload per round:       {avg_round/1024**2:.2f} MB")
        print("=" * 60 + "\n")

    def save_csv(self, path="communication_log_int8.csv"):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "upload_bytes"])
            writer.writeheader()
            writer.writerows(self.round_log)
        print(f"Communication log saved to {path}")


comm_tracker = CommunicationTracker()

# ============================================================
#  int8 quantization helpers (per-tensor symmetric scale)
# ============================================================
def quantize_to_int8(ndarrays):
    """
    Per-tensor symmetric quantization: each tensor gets its own scale
    (max_abs / 127), values mapped to the int8 range [-127, 127].

    Why per-tensor and not a single global scale: parameter tensors in
    this model span very different magnitude ranges (e.g. LayerNorm
    weights vs. attention projection weights). A single global scale
    would waste most of int8's precision on whichever tensor happens to
    have the largest values, crushing everything else toward zero.
    Per-tensor scaling costs a few extra bytes (one float32 per tensor)
    but keeps each tensor's own value range using the full int8 span.

    Returns: quantized int8 arrays + one extra float32 array of scales
    appended at the end (order matches the input list, since
    state_dict().items() iterates in a fixed, consistent order).
    """
    quantized = []
    scales = []
    for i, arr in enumerate(ndarrays):
        arr = arr.astype(np.float32)
        if not np.all(np.isfinite(arr)):
            print(f"WARNING: non-finite values in param tensor {i}, shape {arr.shape}")
        max_abs = np.max(np.abs(arr))
        scale = (max_abs / 127.0) if max_abs > 0 else 1.0
        q = np.round(arr / scale)
        q = np.nan_to_num(q, nan=0.0, posinf=127.0, neginf=-127.0)  # catch any NaN/inf before cast
        q = np.clip(q, -127, 127).astype(np.int8)
        quantized.append(q)
        scales.append(scale)

    scales_arr = np.array(scales, dtype=np.float32)
    quantized.append(scales_arr)
    return quantized


def dequantize_from_int8(ndarrays):
    """Reverses quantize_to_int8: last array is the per-tensor scales,
    everything before it is the quantized int8 tensors in the same order."""
    scales_arr = ndarrays[-1]
    quantized = ndarrays[:-1]

    dequantized = []
    for arr, scale in zip(quantized, scales_arr):
        dequantized.append(arr.astype(np.float32) * scale)
    return dequantized

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
        # quantize to int8 before this leaves the client (upload compression)
        fp32_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return quantize_to_int8(fp32_params)

    def set_parameters(self, parameters):
        if hasattr(parameters, "tensors"):
            params = parameters_to_ndarrays(parameters)
        else:
            params = parameters
        # whatever arrived on the wire (int8 + scales from server),
        # dequantize back to fp32 before loading into the model —
        # training must stay in fp32 for stable gradients.
        params = dequantize_from_int8(params)
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, fit_config):
        self.set_parameters(parameters)

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
                # Prevents exploding gradients from producing NaN weights.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                total_loss += loss.item()

        _t0 = time.perf_counter()
        updated_params = self.get_parameters()  
        transform_time_sec = time.perf_counter() - _t0

        upload_bytes = compute_ndarrays_size(updated_params)
        # what this same delta would have cost at full fp32 precision --
        # dequantize (int8 + scales) back to fp32 and measure that size.
        comm_no_compression_bytes = compute_ndarrays_size(dequantize_from_int8(updated_params))

        return updated_params, len(self.train_loader.dataset), {
            "train_loss": total_loss / len(self.train_loader),
            "upload_bytes": upload_bytes,
            "comm_dense_bytes": upload_bytes,
            "comm_no_compression_bytes": comm_no_compression_bytes,
            "transform_time_sec": transform_time_sec,
        }

    def evaluate(self, parameters, config):
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
    def __init__(self, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0

        self.total_upload_bytes = 0
        self.total_comm_dense_bytes = 0
        self.total_comm_no_compression_bytes = 0
        self.total_transform_time_sec = 0.0
        self.total_reconstruct_time_sec = 0.0

        # Per-round history -- populated in aggregate_fit(), consumed by
        # save_run_data(). 
        self.history = {
            "round": [],
            "accuracy": [],
            "num_bits": [],
            "comm_dense_bytes": [],
            "comm_no_compression_bytes": [],
            "upload_bytes": [],
        }

    def configure_fit(self, server_round, parameters, client_manager):
        # quantize the outgoing global model to int8 before it's
        # distributed to clients.
        ndarrays = parameters_to_ndarrays(parameters)
        quantized_ndarrays = quantize_to_int8(ndarrays)
        quantized_parameters = ndarrays_to_parameters(quantized_ndarrays)
        fit_ins_list = super().configure_fit(server_round, quantized_parameters, client_manager)
        return fit_ins_list

    def configure_evaluate(self, server_round, parameters, client_manager):
        ndarrays = parameters_to_ndarrays(parameters)
        quantized_ndarrays = quantize_to_int8(ndarrays)
        quantized_parameters = ndarrays_to_parameters(quantized_ndarrays)
        return super().configure_evaluate(server_round, quantized_parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round

        # sum up actual upload bytes reported by each client this round
        # (already reflects int8+scale wire size via compute_ndarrays_size)
        round_upload = 0
        round_comm_dense_bytes = 0
        round_comm_no_compression_bytes = 0
        round_transform_time_sec = []
        round_reconstruct_time_sec = 0.0
        for _, fit_res in results:
            metrics = fit_res.metrics
            round_upload += metrics.get("upload_bytes", 0)
            round_comm_dense_bytes += metrics.get("comm_dense_bytes", 0)
            round_comm_no_compression_bytes += metrics.get("comm_no_compression_bytes", 0)
            round_transform_time_sec.append(metrics.get("transform_time_sec", 0.0))
        comm_tracker.log_round(server_round, round_upload)

        # dequantize each client's returned int8 parameters back to
        # fp32 BEFORE FedAvg's weighted averaging runs. Averaging int8
        # values directly (or naively averaging without undoing each
        # client's own per-tensor scale) would produce meaningless
        # results, since each client may have quantized with a different
        # scale. Only the WIRE format was int8; aggregation math stays fp32.
        # Timed the same way FGMP times its server-side decode, for parity.
        _recon_start = time.perf_counter()
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            ndarrays_fp32 = dequantize_from_int8(ndarrays)
            fit_res.parameters = ndarrays_to_parameters(ndarrays_fp32)
        round_reconstruct_time_sec += time.perf_counter() - _recon_start

        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None:
            return aggregated

        parameters, _ = aggregated
        params_ndarrays = parameters_to_ndarrays(parameters)

        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.global_model.state_dict().keys(), params_ndarrays)
        }
        self.global_model.load_state_dict(state_dict, strict=True)

        acc, precision_w, recall_w, f1_w, precision_m, recall_m, f1_m, auc_macro = self.evaluate_global(final=False)

        self.total_upload_bytes += round_upload
        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_no_compression_bytes += round_comm_no_compression_bytes
        avg_transform_time = float(np.mean(round_transform_time_sec)) if round_transform_time_sec else 0.0
        self.total_transform_time_sec += avg_transform_time
        self.total_reconstruct_time_sec += round_reconstruct_time_sec

        # Record this round in our own history dict (see __init__).
        self.history["round"].append(server_round)
        self.history["accuracy"].append(acc)
        self.history["num_bits"].append(8)
        self.history["comm_dense_bytes"].append(round_comm_dense_bytes)
        self.history["comm_no_compression_bytes"].append(round_comm_no_compression_bytes)
        self.history["upload_bytes"].append(round_upload)

        compression_vs_baseline = (
            round_comm_dense_bytes / round_comm_no_compression_bytes
            if round_comm_no_compression_bytes else 1.0
        )
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | num_bits=8")
        print(f"  [comm] ACTUALLY SENT: {round_comm_dense_bytes/1e6:.3f} MB "
              f"({compression_vs_baseline*100:.1f}% of no-compression baseline: {round_comm_no_compression_bytes/1e6:.3f} MB)")
        print(f"  [compute] avg client transform_time: {avg_transform_time*1000:.2f}ms | "
              f"server reconstruct_time: {round_reconstruct_time_sec*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_int8.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            comm_tracker.summary()
            comm_tracker.save_csv("communication_log_int8.csv")
            print(f"Best accuracy achieved: {self.best_acc:.4f}")
            self.save_run_data()

        return aggregated

    def save_run_data(self, path="fl_run_history_int8.npz"):
        h = self.history
        save_kwargs = {k: np.array(v) for k, v in h.items()}

        save_kwargs.update({
            "final_labels": getattr(self, "_final_all_labels", np.array([])),
            "final_preds": getattr(self, "_final_all_preds", np.array([])),
            "final_probs": getattr(self, "_final_all_probs", np.array([])),
            "total_upload_bytes": self.total_upload_bytes,
            "total_comm_dense_bytes": self.total_comm_dense_bytes,
            "total_comm_no_compression_bytes": self.total_comm_no_compression_bytes,
            "total_transform_time_sec": self.total_transform_time_sec,
            "total_reconstruct_time_sec": self.total_reconstruct_time_sec,
            "best_acc": self.best_acc,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "use_compression": True,
            "num_bits_start": 8,
            "num_bits_end": 8,
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

        cm = confusion_matrix(all_labels, all_preds)

        print("\nConfusion Matrix")
        print(cm)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Final Confusion Matrix (int8 quantized comm)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_int8.png")
        plt.close()

        self._final_all_labels = all_labels
        self._final_all_preds = all_preds
        self._final_all_probs = all_probs

        return accuracy

# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset, NUM_CLIENTS, seed=42)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    _tmp_model = IMUTransformerEncoder(config).to(DEVICE)
    print_model_size_summary(_tmp_model)

    # build genuine fp32 initial parameters here, so Flower never falls
    # back to asking a client for get_parameters() (which always returns
    # int8+scales) to seed round 1.
    initial_ndarrays = [val.cpu().numpy() for _, val in _tmp_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)
    del _tmp_model

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
        initial_parameters=initial_parameters,  # passed through **kwargs to FedAvg
    )

    print(f"Starting FL (int8 quantized communication) | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
    )

if __name__ == "__main__":
    main("train.csv", "test.csv")