import flwr as fl
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import warnings
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1

print(f"Using device: {DEVICE}")

# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"],
                               config["input_dim"], config["window_shift"])
    test_dataset = IMUDataset(test_csv, config["window_size"],
                              config["input_dim"], config["window_shift"])
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset

def split_train_data(train_dataset, num_clients=NUM_CLIENTS):
    n = len(train_dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    client_datasets = []
    size = n // num_clients
    for i in range(num_clients):
        start = i * size
        end = start + size if i < num_clients - 1 else n
        subset = Subset(train_dataset, indices[start:end])
        client_datasets.append(subset)
    return client_datasets

# ====================== CLIENT ======================
class IMUClient(fl.client.NumPyClient):
    def __init__(self, train_subset):
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"],
                                       shuffle=True, num_workers=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config["lr"],
                                          weight_decay=config.get("weight_decay", 1e-4))
        self.criterion = torch.nn.NLLLoss()

    def get_parameters(self, config=None):
        """Return parameters as list of numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Accept either list or Parameters object"""
        if hasattr(parameters, "tensors"):  # Flower Parameters object
            params_ndarrays = parameters_to_ndarrays(parameters)
        else:  # raw list
            params_ndarrays = parameters

        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params_ndarrays)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
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

        return self.get_parameters(), len(self.train_loader.dataset), {"train_loss": total_loss / len(self.train_loader)}

# ====================== GLOBAL EVALUATION ======================
def evaluate_global(model, test_loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            imu = batch["imu"].to(DEVICE).float()
            labels = batch["label"].to(DEVICE).long()

            outputs = model({"imu": imu})
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        zero_division=0
    )

    return accuracy, precision, recall, f1, cm, report

# ====================== STRATEGY ======================
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.best_acc = 0.0
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n📊 Round {server_round} | {len(results)} successful clients, {len(failures)} failures")

        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            print("Aggregation failed")
            return aggregated

        parameters, _ = aggregated
        params_ndarrays = parameters_to_ndarrays(parameters)

        # Load global model
        state_dict = {k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), params_ndarrays)}
        self.global_model.load_state_dict(state_dict, strict=True)

        accuracy, precision, recall, f1, cm, report = evaluate_global(
        self.global_model,
        self.test_loader
        )

        print("\n==============================")
        print(f"Round {server_round}")
        print("==============================")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report")
        print(report)

        print("Confusion Matrix")
        print(cm)

        if accuracy > self.best_acc:
            self.best_acc = accuracy

            torch.save(
                self.global_model.state_dict(),
                f"best_global_model_r{server_round}.pth"
            )

            print("Best model saved!")

            plt.figure(figsize=(8,6))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues"
            )

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix Round {server_round}")

            plt.savefig("best_confusion_matrix.png")
            plt.close()

            with open("classification_report.txt", "w") as f:
                f.write(report)
        return aggregated

# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    def client_fn(cid: str):
        return IMUClient(client_datasets[int(cid)]).to_client()

    strategy = SaveModelStrategy(
        test_loader=test_loader,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    print(f"Starting FL | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.1 if torch.cuda.is_available() else 0},
    )

    print(f"\nFinished! Best Accuracy: {strategy.best_acc:.4f}")

if __name__ == "__main__":
    main("train.csv", "test.csv")