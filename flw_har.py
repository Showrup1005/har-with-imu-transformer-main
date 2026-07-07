import flwr as fl
import torch
import numpy as np
import json
import warnings
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from torch.utils.data import DataLoader
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
    test_dataset = IMUDataset(
        test_csv, 
        config["window_size"], 
        config["input_dim"], 
        config["window_shift"]
    )
    print(f"Test samples: {len(test_dataset)}")
    return None, test_dataset

def split_train_data(train_csv, num_clients=NUM_CLIENTS, 
                     save_file="subject_split.json", seed=42):
    df = pd.read_csv(train_csv)
    subjects = sorted(df["subject"].unique())
    print(f"Total Subjects: {len(subjects)}")

    if os.path.exists(save_file):
        print(f"Loading existing subject split from {save_file}")
        with open(save_file, "r") as f:
            client_subjects = json.load(f)
    else:
        print(f"Creating new subject split and saving to {save_file}")
        np.random.seed(seed)
        np.random.shuffle(subjects)
        groups = np.array_split(subjects, num_clients)
        client_subjects = {str(i): [int(s) for s in groups[i]] for i in range(num_clients)}
        with open(save_file, "w") as f:
            json.dump(client_subjects, f, indent=4)

    client_datasets = []
    for cid in range(num_clients):
        group = client_subjects[str(cid)]
        client_ds = IMUDataset(
            train_csv,
            config["window_size"],
            config["input_dim"],
            config["window_shift"],
            subject_ids=group
        )
        client_datasets.append(client_ds)
        print(f"Client {cid} → {len(group)} subjects | {len(client_ds)} windows")
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
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if hasattr(parameters, "tensors"):
            params_ndarrays = parameters_to_ndarrays(parameters)
        else:
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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            imu = batch["imu"].to(DEVICE).float()
            labels = batch["label"].to(DEVICE).long()
            outputs = model({"imu": imu})
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    return accuracy, precision, recall, f1, cm, report

# ====================== STRATEGY ======================
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.model_size = sum(p.numel() * p.element_size() for p in self.global_model.parameters())
        self.total_upload = self.total_download = self.total_comm_time = 0

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated

        parameters, _ = aggregated
        params_ndarrays = parameters_to_ndarrays(parameters)
        state_dict = {k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), params_ndarrays)}
        self.global_model.load_state_dict(state_dict)

        # Global Evaluation
        accuracy, precision, recall, f1, cm, report = evaluate_global(self.global_model, self.test_loader)
        print(f"\nRound {server_round} - Global Accuracy: {accuracy:.4f}")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL RESULTS ==========")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print("\nClassification Report:\n", report)
            torch.save(self.global_model.state_dict(), "final_global_model.pth")

        return aggregated

# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    _, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_csv, NUM_CLIENTS)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    def client_fn(context):
        if hasattr(context, "node_id"):
            cid = int(context.node_id)
        else:
            cid = int(context)
        
        # Critical safety fix
        if cid >= len(client_datasets):
            cid = cid % len(client_datasets)
        
        print(f"Client {cid} requested")  # for debugging
        return IMUClient(client_datasets[cid]).to_client()

    strategy = SaveModelStrategy(test_loader=test_loader)

    print(f"Starting FL | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.1 if torch.cuda.is_available() else 0},
    )


if __name__ == "__main__":
    main("train.csv", "test.csv")