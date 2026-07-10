import flwr as fl
import torch
import numpy as np
import warnings
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from util.IMUPreprocess import IMUPreprocessor
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 3
MU = 0.01
PROTO_WEIGHT = 0.1

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

        # === Label Distribution ===
        labels = [sample['label'].item() if torch.is_tensor(sample['label']) else sample['label'] for sample in client_ds]
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"    Label distribution: {dist}")
        print(f"    Total windows: {sum(counts)}")

    return client_datasets

# ====================== CLIENT ======================
class IMUClient(fl.client.NumPyClient):
    def __init__(self, train_subset, client_id):
        self.client_id = client_id
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"] * 0.5)
        self.criterion = torch.nn.NLLLoss()
        self.preprocessor = IMUPreprocessor(fs=50.0)  

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

        global_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        for _ in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                batch["imu"] = imu  # ensure key exists
                batch = self.preprocessor(batch)
                imu = batch["imu"]
                labels = batch["label"].to(DEVICE).long()

                features = self.model.get_features({"imu": imu})
                output = self.model({"imu": imu})

                ce_loss = self.criterion(output, labels)

                # Prototype Loss
                proto_loss = 0.0
                if len(labels) > 1:
                    unique_labels = torch.unique(labels)
                    for lbl in unique_labels:
                        mask = (labels == lbl)
                        if mask.sum() > 1:
                            class_features = features[mask]
                            prototype = class_features.mean(dim=0)
                            proto_loss += torch.norm(class_features - prototype, p=2, dim=1).mean()

                proto_loss = proto_loss / max(1, len(unique_labels))

                # FedProx Term
                prox_loss = 0.0
                for k, v in self.model.state_dict().items():
                    if k in global_state:
                        prox_loss += torch.norm(v - global_state[k], p=2)**2
                prox_loss = MU * prox_loss

                loss = ce_loss + PROTO_WEIGHT * proto_loss + prox_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += ce_loss.item()

        return self.get_parameters(), len(self.train_loader.dataset), {"train_loss": total_loss / len(self.train_loader)}

    def evaluate(self, parameters, config):
        # Minimal dummy to prevent failures
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {"accuracy": 0.0}

# ====================== STRATEGY ======================
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.preprocessor = IMUPreprocessor(fs=50.0)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated

        parameters, _ = aggregated
        params_ndarrays = parameters_to_ndarrays(parameters)
        state_dict = {k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), params_ndarrays)}
        self.global_model.load_state_dict(state_dict)

        accuracy = self.evaluate_global()
        print(f"\nRound {server_round} - Global Accuracy: {accuracy:.4f}")

        if server_round == NUM_ROUNDS:
            print(f"Final Accuracy: {accuracy:.4f}")
            torch.save(self.global_model.state_dict(), "final_global_model.pth")

        return aggregated

    def evaluate_global(self):
        self.global_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                imu = batch["imu"].to(DEVICE).float()
                batch["imu"] = imu  # ensure key exists
                batch = self.preprocessor(batch)
                imu = batch["imu"]
                labels = batch["label"].to(DEVICE).long()
                output = self.global_model({"imu": imu})
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return accuracy_score(all_labels, all_preds)

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