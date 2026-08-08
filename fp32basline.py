import flwr as fl
import torch
import numpy as np
import json
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
NUM_ROUNDS = 60

print(f"Using device: {DEVICE}")

# ============================================================
#  communication cost measurement utilities
#
#  IMPORTANT: your SAC and FedZip scripts only ever measure the
#  client->server UPLOAD (delta = new_val - old_val, computed inside
#  fit(), nothing about the incoming `parameters` is measured). This
#  script now tracks both directions but treats UPLOAD-ONLY as the
#  number to put in a comparison table against SAC/FedZip -- that's
#  also the convention the FedZip paper itself uses (their Table 1
#  labels every method, including FedAvg, "Upstream"). The full
#  round-trip total is still printed/saved for reference, just kept
#  clearly separate so it doesn't get mixed into a comparison it
#  isn't comparable to.
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

    print("\n" + "=" * 60)
    print("MODEL SIZE SUMMARY")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (fp32):    {size_bytes:,} bytes = {size_mb:.2f} MB")
    print("=" * 60 + "\n")
    return size_bytes


class CommunicationTracker:
    """Accumulates real transmitted bytes across the whole FL run:
    download = server -> client (parameters sent into fit)
    upload   = client -> server (parameters returned from fit)

    upload-only is the figure comparable to SAC/FedZip's
    comm_dense_bytes; download+upload is the full round-trip cost,
    reported separately so the two are never silently mixed.
    """
    def __init__(self):
        self.round_log = []  # list of dicts per round
        self.total_download_bytes = 0
        self.total_upload_bytes = 0

    def log_round(self, server_round, download_bytes, upload_bytes):
        self.total_download_bytes += download_bytes
        self.total_upload_bytes += upload_bytes
        total_round_bytes = download_bytes + upload_bytes
        self.round_log.append({
            "round": server_round,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "total_bytes": total_round_bytes,
        })
        print(f"  [Comm] Round {server_round}: "
              f"download={download_bytes/1024**2:.2f} MB, "
              f"upload={upload_bytes/1024**2:.2f} MB (<- comparable to SAC/FedZip), "
              f"round_total={total_round_bytes/1024**2:.2f} MB")

    def summary(self):
        total = self.total_download_bytes + self.total_upload_bytes
        print("\n" + "=" * 60)
        print("COMMUNICATION COST SUMMARY")
        print("=" * 60)
        print(f"Total download (server->client):        {self.total_download_bytes/1024**2:.2f} MB")
        print(f"Total upload   (client->server):         {self.total_upload_bytes/1024**2:.2f} MB   <-- use this for the SAC/FedZip comparison table")
        print(f"Total round-trip (download + upload):    {total/1024**2:.2f} MB ({total/1024**3:.4f} GB)")
        if self.round_log:
            avg_round_upload = self.total_upload_bytes / len(self.round_log)
            avg_round_total = total / len(self.round_log)
            print(f"Average upload per round:                {avg_round_upload/1024**2:.2f} MB")
            print(f"Average round-trip per round:             {avg_round_total/1024**2:.2f} MB")
        print("=" * 60 + "\n")

    def save_csv(self, path="communication_log.csv"):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "download_bytes", "upload_bytes", "total_bytes"])
            writer.writeheader()
            writer.writerows(self.round_log)
        print(f"Communication log saved to {path}")


comm_tracker = CommunicationTracker()

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

    def fit(self, parameters, config):
        # measure download size (server -> this client) before training
        download_bytes = compute_ndarrays_size(parameters)

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
                self.optimizer.step()
                total_loss += loss.item()

        updated_params = self.get_parameters()
        # measure upload size (this client -> server) -- this is the
        # figure directly comparable to SAC/FedZip's comm_dense_bytes
        upload_bytes = compute_ndarrays_size(updated_params)

        return updated_params, len(self.train_loader.dataset), {
            "train_loss": total_loss / len(self.train_loader),
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
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

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round

        # sum up actual download/upload bytes reported by each client this round
        round_download = 0
        round_upload = 0
        for _, fit_res in results:
            metrics = fit_res.metrics
            round_download += metrics.get("download_bytes", 0)
            round_upload += metrics.get("upload_bytes", 0)
        comm_tracker.log_round(server_round, round_download, round_upload)

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

        # Accuracy after every round
        acc = self.evaluate_global(final=False)
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f}")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model.pth")

        # Final evaluation after last round
        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            # print and save communication summary at the very end
            comm_tracker.summary()
            comm_tracker.save_csv("communication_log.csv")

        return aggregated

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

        # During training rounds, return only accuracy
        if not final:
            return accuracy

        # Only after final round
        precision = precision_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0,
        )

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
        plt.title("Final Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix.png")
        plt.close()

        return accuracy

# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset, NUM_CLIENTS, seed=42)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # print model size / theoretical per-round cost before training starts
    _tmp_model = IMUTransformerEncoder(config).to(DEVICE)
    print_model_size_summary(_tmp_model)
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

    strategy = SaveModelStrategy(test_loader=test_loader)

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