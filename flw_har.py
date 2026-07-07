import flwr as fl
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import warnings
import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import time
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from torch.utils.data import Subset
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

# def split_train_data(train_dataset, num_clients=NUM_CLIENTS):
#     n = len(train_dataset)
#     indices = np.arange(n)
#     np.random.shuffle(indices)
#     client_datasets = []
#     size = n // num_clients
#     for i in range(num_clients):
#         start = i * size
#         end = start + size if i < num_clients - 1 else n
#         subset = Subset(train_dataset, indices[start:end])
#         client_datasets.append(subset)
#     return client_datasets

def split_train_data(train_dataset,
                     train_csv,
                     num_clients=NUM_CLIENTS,
                     save_file="subject_split.json"):

    df = pd.read_csv(train_csv)

    subjects = sorted(df["subject"].unique())

    print(f"Total Subjects: {len(subjects)}")

    # ----------------------------
    # Load existing split
    # ----------------------------
    if os.path.exists(save_file):

        print("Loading existing subject split...")

        with open(save_file, "r") as f:
            client_subjects = json.load(f)

    # ----------------------------
    # Create split once
    # ----------------------------
    else:

        np.random.seed(42)

        np.random.shuffle(subjects)

        groups = np.array_split(subjects, num_clients)

        client_subjects = {
            str(i): list(map(int, groups[i]))
            for i in range(num_clients)
        }

        with open(save_file, "w") as f:
            json.dump(client_subjects, f, indent=4)

        print("Subject split saved.")

    client_datasets = []

    for cid in range(num_clients):

        group = client_subjects[str(cid)]

        indices = df[df["subject"].isin(group)].index.tolist()

        subset = Subset(train_dataset, indices)

        client_datasets.append(subset)

        print("\n--------------------------------")
        print(f"Client {cid}")
        print(f"Subjects : {group}")
        print(f"Samples  : {len(indices)}")

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

def get_model_size_bytes(model):
    """Returns model size in bytes."""
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, **kwargs):
        super().__init__(**kwargs)

        self.test_loader = test_loader

        self.best_acc = 0.0

        self.global_model = IMUTransformerEncoder(config).to(DEVICE)

        # Communication statistics
        self.round_start_time = 0
        self.total_comm_time = 0

        self.model_size = get_model_size_bytes(self.global_model)

        self.total_upload = 0
        self.total_download = 0

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.perf_counter()

        return super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None:
            return aggregated

        parameters, _ = aggregated

        params_ndarrays = parameters_to_ndarrays(parameters)

        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(
                self.global_model.state_dict().keys(),
                params_ndarrays,
            )
        }

        self.global_model.load_state_dict(state_dict)

        # -------------------------------
        # Communication statistics
        # -------------------------------

        round_time = time.perf_counter() - self.round_start_time

        self.total_comm_time += round_time

        upload = self.model_size * NUM_CLIENTS
        download = self.model_size * NUM_CLIENTS

        self.total_upload += upload
        self.total_download += download

        total_comm = self.total_upload + self.total_download

        print(f"\nRound {server_round}")

        print(f"Communication Time : {round_time:.3f} sec")

        print(f"Upload            : {upload/1024/1024:.2f} MB")

        print(f"Download          : {download/1024/1024:.2f} MB")

        print(f"Total Comm.       : {total_comm/1024/1024:.2f} MB")

        accuracy, precision, recall, f1, cm, report = evaluate_global(
                    self.global_model,
                    self.test_loader,
                )
    
        print("\n==============================")
        print(f"Round {server_round}")
        print("==============================")

        print(f"Global Accuracy : {accuracy:.4f}")

        if server_round == NUM_ROUNDS:

            print("\n========== FINAL RESULTS ==========")

            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            print("\nClassification Report")
            print(report)

            print("\nConfusion Matrix")
            print(cm)

            plt.figure(figsize=(8,6))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues"
            )

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Final Confusion Matrix")

            plt.savefig("final_confusion_matrix.png")

            plt.close()

            with open("classification_report.txt","w") as f:
                f.write(report)

            torch.save(
                self.global_model.state_dict(),
                "final_global_model.pth"
            )

            print("\n=========== Communication Summary ===========")

            print(f"Model Size          : {self.model_size/1024/1024:.2f} MB")

            print(f"Total Upload        : {self.total_upload/1024/1024:.2f} MB")

            print(f"Total Download      : {self.total_download/1024/1024:.2f} MB")

            print(f"Communication Total : {(self.total_upload+self.total_download)/1024/1024:.2f} MB")

            print(f"Communication Time  : {self.total_comm_time:.3f} sec")

        return aggregated

# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(
        train_dataset,
        train_csv,
        NUM_CLIENTS
    )   

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


if __name__ == "__main__":
    main("train.csv", "test.csv")