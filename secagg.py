"""
Federated HAR training with Secure Aggregation (SecAgg).

This is a DIFFERENT CATEGORY of privacy mechanism than SAPM/DP-FedAvg:
those work by removing or corrupting information before transmission
(hence their accuracy cost). SecAgg works by CRYPTOGRAPHICALLY HIDING
each client's individual update from the server while still letting the
server compute the exact correct sum -- no information is lost, so
accuracy should come out essentially identical to plain FedAvg. The
"cost" of this mechanism is computational/communication overhead and a
narrower threat-model guarantee, not utility.

HOW IT WORKS (real X25519 Elliptic-Curve Diffie-Hellman, not simulated):
    1. Once at startup, every pair of clients (i, j) independently derives
       a shared secret via ECDH. The server only ever relays each
       client's PUBLIC key to the others -- by the math of Diffie-Hellman,
       relaying public keys does not reveal the shared secret to anyone
       who doesn't hold one of the two private keys, so the server
       genuinely cannot derive these secrets itself.
    2. Each round, every pair's shared secret is used to seed a
       pseudorandom mask vector (same size as the flattened model).
       Client i ADDS this mask if i < j, and SUBTRACTS it if i > j, for
       every other client j.
    3. Client i sends (its true update + the sum of all its pairwise
       masks) to the server. Individually, this masked update is
       indistinguishable from random noise to the server -- it carries no
       usable information about the client's true update.
    4. When the server sums ALL clients' masked updates together, every
       pairwise mask appears exactly once with +1 and once with -1 (once
       from each side of the pair), so they cancel exactly. The server is
       left with the exact true sum, with zero information loss --
       mathematically identical to what it would compute from plaintext
       updates.

WHAT THIS DOES vs. DOES NOT PROTECT AGAINST:
    - DOES prevent the server from ever seeing any individual client's
      raw update -- this is the exact single-client gradient-inversion
      attack tested throughout this project (the one that leaked labels
      via the bias-gradient sign). Under SecAgg, that attack has no
      individual update to target at all.
    - Does NOT prevent the server from attempting reconstruction attacks
      against the AGGREGATE (the sum across all NUM_CLIENTS clients) it
      legitimately receives every round -- that's the same plaintext
      aggregate a non-private FedAvg server would see. With only 3
      clients (vs. the hundreds/thousands typical in production SecAgg
      deployments), this aggregate is a much smaller "anonymity set" than
      real-world usage, which is worth stating as a limitation.

SIMPLIFICATIONS vs. production SecAgg (Bonawitz et al. 2017), stated
explicitly rather than hidden:
    - No dropout resilience (no Shamir secret-sharing fallback) -- all
      NUM_CLIENTS clients are assumed to always participate every round.
      Production SecAgg handles clients dropping mid-protocol; this demo
      does not.
    - Aggregation uses SIMPLE (unweighted) averaging across clients,
      because the pairwise-mask cancellation math above requires
      symmetric +/- contributions -- example-count-weighted averaging
      would need masks scaled by weight too (a known extension, not
      implemented here). Since your 3 clients have almost identical
      partition sizes (~8,161 each), the difference from weighted
      averaging is negligible in practice.
    - The raw ECDH shared secret is passed through HKDF (a standard KDF)
      before use as a PRG seed, which is correct practice -- but the PRG
      itself is numpy's RandomState, which is fast but NOT
      cryptographically secure. For a production system you'd want a
      real CSPRNG (e.g. AES-CTR-DRBG); for this simulation (no live
      network adversary) it's a reasonable simplification.
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

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

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
MASK_SCALE = 10.0   # std of the pairwise mask noise; any value works (it
                     # cancels exactly regardless of magnitude), chosen
                     # large relative to typical delta magnitudes (~1-10,
                     # per earlier DP runs) so individual masked updates
                     # carry no visible signal.

print(f"Using device: {DEVICE}")
print("Privacy strategy: Secure Aggregation (real X25519 ECDH pairwise masking, unweighted mean)")


# ====================== KEY EXCHANGE (done once, before training) ======================
def generate_pairwise_secrets(num_clients: int):
    """Real ECDH key exchange between every pair of clients. Returns
    {client_idx: {other_idx: shared_secret_bytes}}. The private keys
    never leave this function -- only public keys would need to cross
    any real network boundary, and even those aren't needed here since
    this is a single-process simulation; we do the full DH math anyway
    (not just handing out a common seed) so the derived secrets are
    exactly as strong as in a real deployment."""
    private_keys = [X25519PrivateKey.generate() for _ in range(num_clients)]
    public_keys = [pk.public_key() for pk in private_keys]

    secrets = {i: {} for i in range(num_clients)}
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                continue
            raw_shared = private_keys[i].exchange(public_keys[j])
            # HKDF: standard practice to turn a raw DH output into a
            # uniformly-random key suitable for use as a PRG seed.
            derived = HKDF(
                algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"secagg-mask-seed",
            ).derive(raw_shared)
            secrets[i][j] = derived
    return secrets


def mask_seed_for_round(shared_secret: bytes, round_num: int) -> int:
    """Deterministic per-round seed derived from a pairwise shared
    secret. Both sides of the pair compute this identically without
    further communication."""
    h = hashes.Hash(hashes.SHA256())
    h.update(shared_secret)
    h.update(round_num.to_bytes(4, "big"))
    digest = h.finalize()
    return int.from_bytes(digest[:4], "big")


def generate_mask(shared_secret: bytes, round_num: int, size: int, scale: float) -> np.ndarray:
    seed = mask_seed_for_round(shared_secret, round_num)
    rng = np.random.RandomState(seed % (2 ** 31 - 1))
    return rng.normal(0.0, scale, size=size).astype(np.float32)


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
        print(f"Client {i} -> {len(subset)} samples")
    print("=" * 60)
    return client_datasets


# ====================== CLIENT ======================
class IMUClient(fl.client.NumPyClient):
    def __init__(self, train_subset, client_idx, pairwise_secrets):
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.client_idx = client_idx
        self.pairwise_secrets = pairwise_secrets  # {other_idx: shared_secret_bytes}

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
        round_num = fit_config.get("server_round", 0)

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
        keys = list(new_state.keys())
        deltas = [(new_state[k] - old_state[k]).cpu().numpy().astype(np.float32) for k in keys]
        shapes = [d.shape for d in deltas]
        flat = np.concatenate([d.reshape(-1) for d in deltas])

        # ---- SecAgg masking ----
        mask_start = time.time()
        combined_mask = np.zeros_like(flat)
        for other_idx, secret in self.pairwise_secrets.items():
            m = generate_mask(secret, round_num, flat.size, MASK_SCALE)
            combined_mask += m if self.client_idx < other_idx else -m
        masked_flat = flat + combined_mask
        mask_time = time.time() - mask_start

        # split back into per-tensor arrays matching the original shapes
        out_arrays = []
        offset = 0
        for shape in shapes:
            n = int(np.prod(shape))
            out_arrays.append(masked_flat[offset:offset + n].reshape(shape).astype(np.float32))
            offset += n

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "mask_time_sec": mask_time,
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
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["server_round"] = server_round
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        # UNWEIGHTED sum of masked updates -- required for exact mask
        # cancellation (see module docstring). The server never sees any
        # individual client's true delta, only ever this sum.
        summed_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        mask_times = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            mask_times.append(fit_res.metrics.get("mask_time_sec", 0.0))
            for k, arr in zip(keys, arrays):
                summed_deltas[k] += arr.astype(np.float64)

        n_clients = len(results)
        new_state = {}
        for k in keys:
            avg_delta = summed_deltas[k] / n_clients  # masks cancelled exactly in the sum above
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype, device=global_state[k].device)

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_mask_time = float(np.mean(mask_times)) if mask_times else 0.0
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | Avg client-side mask time: {avg_mask_time*1000:.1f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_secagg.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)

        return aggregated_params, {"accuracy": acc, "avg_mask_time_ms": avg_mask_time * 1000}

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
        plt.title("Final Confusion Matrix (Secure Aggregation)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_secagg.png")
        plt.close()
        return accuracy


# ====================== MAIN ======================
def main(train_csv: str, test_csv: str):
    train_dataset, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_dataset, NUM_CLIENTS, seed=42)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    print("Performing real X25519 ECDH key exchange between all client pairs...")
    all_secrets = generate_pairwise_secrets(NUM_CLIENTS)
    print(f"Derived {sum(len(v) for v in all_secrets.values())} directed pairwise secrets "
          f"({NUM_CLIENTS * (NUM_CLIENTS - 1) // 2} unordered pairs) via ECDH.\n")

    def client_fn(context):
        if hasattr(context, "node_id"):
            cid = int(context.node_id)
        elif hasattr(context, "node_config") and "cid" in context.node_config:
            cid = int(context.node_config["cid"])
        else:
            cid = 0
        client_idx = cid % len(client_datasets)
        return IMUClient(client_datasets[client_idx], client_idx, all_secrets[client_idx]).to_client()

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