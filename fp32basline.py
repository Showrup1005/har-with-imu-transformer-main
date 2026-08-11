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

STABILITY_LAMBDA = 0.01   

print(f"Using device: {DEVICE}")
print(f"Strategy: FP32, regularized (no compression) | stability_lambda={STABILITY_LAMBDA}")

# ============================================================
#  communication cost measurement utilities
# ============================================================
def compute_ndarrays_size(ndarrays):
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
    def __init__(self):
        self.round_log = []
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
              f"upload={upload_bytes/1024**2:.2f} MB, "
              f"round_total={total_round_bytes/1024**2:.2f} MB")

    def summary(self):
        total = self.total_download_bytes + self.total_upload_bytes
        print("\n" + "=" * 60)
        print("COMMUNICATION COST SUMMARY")
        print("=" * 60)
        print(f"Total download (server->client):        {self.total_download_bytes/1024**2:.2f} MB")
        print(f"Total upload   (client->server):         {self.total_upload_bytes/1024**2:.2f} MB")
        print(f"Total round-trip (download + upload):    {total/1024**2:.2f} MB ({total/1024**3:.4f} GB)")
        if self.round_log:
            avg_round_upload = self.total_upload_bytes / len(self.round_log)
            avg_round_total = total / len(self.round_log)
            print(f"Average upload per round:                {avg_round_upload/1024**2:.2f} MB")
            print(f"Average round-trip per round:             {avg_round_total/1024**2:.2f} MB")
        print("=" * 60 + "\n")

    def save_csv(self, path="communication_log_fp32_reg.csv"):
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

    def fit(self, parameters, fit_config):
        download_bytes = compute_ndarrays_size(parameters)

        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}
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

        for _ in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()

                self.optimizer.zero_grad()
                output = self.model({"imu": imu})
                ce_loss = self.criterion(output, label)

                reg_loss = torch.zeros((), device=DEVICE)
                if stability_lambda > 0 and n_grad_steps > 0:
                    for name, p in self.model.named_parameters():
                        if name in fisher_accum:
                            f_running = (fisher_accum[name] / n_grad_steps).detach()
                            reg_loss = reg_loss + (f_running * (p - old_state[name]) ** 2).sum()

                loss = ce_loss + stability_lambda * reg_loss
                loss.backward()

                for name, p in self.model.named_parameters():
                    if p.grad is not None and name in fisher_accum:
                        fisher_accum[name] += p.grad.detach() ** 2
                n_grad_steps += 1

                self.optimizer.step()
                total_loss += ce_loss.item()
                total_reg_loss += float(reg_loss.detach().item())

        updated_params = self.get_parameters()
        upload_bytes = compute_ndarrays_size(updated_params)

        return updated_params, len(self.train_loader.dataset), {
            "train_loss": total_loss / len(self.train_loader),
            "avg_reg_loss": total_reg_loss / len(self.train_loader),
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
    def __init__(self, test_loader, stability_lambda=STABILITY_LAMBDA, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.stability_lambda = stability_lambda

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["stability_lambda"] = self.stability_lambda
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round

        round_download = 0
        round_upload = 0
        reg_losses = []
        for _, fit_res in results:
            metrics = fit_res.metrics
            round_download += metrics.get("download_bytes", 0)
            round_upload += metrics.get("upload_bytes", 0)
            reg_losses.append(metrics.get("avg_reg_loss", 0.0))
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

        acc = self.evaluate_global(final=False)
        avg_reg = float(np.mean(reg_losses)) if reg_losses else 0.0
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | avg_reg_loss: {avg_reg:.5f}")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_fp32_reg.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            comm_tracker.summary()
            comm_tracker.save_csv("communication_log_fp32_reg.csv")
            print(f"Best accuracy achieved: {self.best_acc:.4f}")

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
        plt.title("Final Confusion Matrix (FP32, regularized, no compression)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_fp32_reg.png")
        plt.close()

        return accuracy

# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset, NUM_CLIENTS, seed=42)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

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

    strategy = SaveModelStrategy(test_loader=test_loader, stability_lambda=STABILITY_LAMBDA)

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