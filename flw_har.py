import flwr as fl
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 3
MU = 0.01                    # FedProx strength
PROTO_WEIGHT = 0.1           # Weight for prototype loss

print(f"Using device: {DEVICE} | FedProx mu={MU} | Proto weight={PROTO_WEIGHT}")

# ====================== DATA ======================
def load_data(train_csv: str, test_csv: str):
    test_dataset = IMUDataset(test_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Test samples: {len(test_dataset)}")
    return None, test_dataset

def split_train_data(train_csv, num_clients=NUM_CLIENTS, save_file="subject_split.json", seed=42):
    df = pd.read_csv(train_csv)
    subjects = sorted(df["subject"].unique())
    print(f"Total Subjects: {len(subjects)}")

    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            client_subjects = json.load(f)
    else:
        np.random.seed(seed)
        np.random.shuffle(subjects)
        groups = np.array_split(subjects, num_clients)
        client_subjects = {str(i): [int(s) for s in groups[i]] for i in range(num_clients)}
        with open(save_file, "w") as f:
            json.dump(client_subjects, f, indent=4)

    client_datasets = []
    for cid in range(num_clients):
        group = client_subjects[str(cid)]
        client_ds = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"], subject_ids=group)
        client_datasets.append(client_ds)
        print(f"Client {cid} → {len(group)} subjects | {len(client_ds)} windows")
    return client_datasets

# ====================== CLIENT ======================
class IMUClient(fl.client.NumPyClient):
    def __init__(self, train_subset, client_id):
        self.client_id = client_id
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"] * 0.5)
        self.criterion = torch.nn.NLLLoss()

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
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0

        for _ in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()

                # Forward
                features = self.model.get_features({"imu": imu})
                output = self.model({"imu": imu})

                # Standard loss
                ce_loss = self.criterion(output, labels)

                # Prototype regularization loss
                proto_loss = 0.0
                for i in range(len(labels)):
                    proto_loss += torch.norm(features[i] - features[labels[i]], p=2)**2   # Simple version
                proto_loss = proto_loss / len(labels)

                loss = ce_loss + PROTO_WEIGHT * proto_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += ce_loss.item()

        return self.get_parameters(), len(self.train_loader.dataset), {"train_loss": total_loss / len(self.train_loader)}


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
        if cid >= len(client_datasets):
            cid = cid % len(client_datasets)
        return IMUClient(client_datasets[cid], cid).to_client()

    strategy = fl.server.strategy.FedAvg(
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
        client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
    )


if __name__ == "__main__":
    main("train.csv", "test.csv")