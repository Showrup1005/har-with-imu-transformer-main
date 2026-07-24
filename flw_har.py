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
NUM_ROUNDS = 40

print(f"Using device: {DEVICE}")

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
    
    print("="*60)
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
