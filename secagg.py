"""
Federated HAR training with Secure Aggregation (SecAgg) + Top-K Sparsification.

THE PROBLEM THIS FILE ADDS ON TOP OF secagg.py:
    Plain SecAgg (secagg.py) sends a masked vector the same size as the
    FULL model every round. That's fine for a tiny model but becomes the
    communication bottleneck for large ones. Top-K sparsification fixes
    this by having each client send only its K largest-magnitude update
    entries instead of all D of them.

    The catch (this is the exact issue raised in "Secure Aggregation with
    Top-K Sparsification in Decentralized Federated Learning", Tang,
    Zhu & Tang 2026, and independently in "Secure Aggregation Meets
    Sparsification in Decentralized Learning" (CESAR), 2024): each
    client's own top-K indices are data-dependent, so client i's chosen
    coordinates and client j's chosen coordinates generally don't match.
    SecAgg's pairwise +m/-m cancellation only works when both sides of a
    pair mask the SAME coordinates -- if their supports differ, the
    unmatched mask entries don't cancel and corrupt the sum. Those papers
    solve this with fairly heavy machinery (offline permutation/secret-
    sharing schemes so mismatched supports can still be reconciled
    cryptographically).

THE SIMPLIFICATION USED HERE, STATED EXPLICITLY:
    Instead of reconciling different per-client supports after the fact,
    this implementation avoids the mismatch altogether by making the
    top-K support a single PUBLIC, SERVER-DETERMINED set S that every
    client is told to use for a given round (sent alongside the global
    model in configure_fit, so it costs no extra round trip). Concretely:
        - Round 1 has no aggregate history yet, so S is a fixed random
          K-subset of coordinates (seeded, so every client agrees on it
          without communication).
        - From round 2 onward, S is the top-K coordinates by |magnitude|
          of the PREVIOUS round's aggregated (already-public) update --
          i.e. the same delta the server just wrote into the global
          model. Because that aggregate is public information everyone
          already sees, choosing S from it leaks nothing beyond what
          plain FedAvg already reveals.
    Since every client masks and sends values at the exact same K
    indices, the original pairwise-mask cancellation proof from
    secagg.py applies completely unchanged -- just with vectors of length
    K instead of length D. This is a real accuracy/communication
    trade-off relative to the papers' approach: letting each client keep
    its OWN locally-optimal top-K (rather than a shared, slightly-stale
    one) would track each client's individual large-gradient coordinates
    more precisely. What we keep is the thing that matters most for a
    demo: exact mask cancellation with zero extra cryptographic
    machinery, and genuine communication savings that scale with K/D.

WHAT DOESN'T CHANGE FROM secagg.py:
    - The ECDH key exchange, HKDF derivation, and per-round PRG masking
      math are identical (see generate_pairwise_secrets / generate_mask
      there) -- only the SIZE of the vector being masked changes, from D
      (full model) to K (support size).
    - The server still only ever sees a SUM of masked values, never an
      individual client's true update -- Top-K doesn't weaken that
      guarantee, it only changes how many coordinates are summed.
    - Same limitations apply: no dropout resilience, unweighted
      averaging only, numpy RandomState (not a CSPRNG) as the PRG, and a
      3-client anonymity set that's far smaller than production
      deployments.

NEW LIMITATION INTRODUCED BY SPARSIFICATION:
    - Non-selected coordinates are implicitly treated as zero-delta for
      that round. Because S is refreshed every round based on the latest
      aggregate, a coordinate that matters but was quiet last round can
      still be picked up once it starts moving -- but there will always
      be a 1-round lag between "this coordinate started mattering" and
      "this coordinate got included in S". With a small K this can slow
      convergence relative to full-gradient SecAgg; TOP_K_FRACTION is the
      knob that trades communication savings against that lag.
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
MASK_SCALE = 10.0        # std of the pairwise mask noise (cancels exactly
                          # regardless of magnitude; see secagg.py).
TOP_K_FRACTION = 0.10    # fraction of total model parameters kept each
                          # round. Lower = more communication savings,
                          # more staleness lag (see docstring above).

print(f"Using device: {DEVICE}")
print(f"Privacy strategy: Secure Aggregation (X25519 ECDH pairwise masking) "
      f"+ Top-{TOP_K_FRACTION*100:.0f}% public-support sparsification")


# ====================== KEY EXCHANGE (done once, before training) ======================
def generate_pairwise_secrets(num_clients: int):
    """Real ECDH key exchange between every pair of clients. Identical to
    secagg.py -- unaffected by sparsification, since it just establishes
    per-pair shared secrets, not anything dimension-dependent."""
    private_keys = [X25519PrivateKey.generate() for _ in range(num_clients)]
    public_keys = [pk.public_key() for pk in private_keys]

    secrets = {i: {} for i in range(num_clients)}
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                continue
            raw_shared = private_keys[i].exchange(public_keys[j])
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
    """Same PRG as secagg.py, just now called with size=K (support size)
    instead of size=D (full model). Both sides of a pair call this with
    the identical (shared_secret, round_num, size) triple -- since S is
    public and agreed by construction, size is always identical too, so
    the resulting mask vectors still cancel exactly on summation."""
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


def get_param_layout(model):
    """Flatten a model's state_dict into (keys, shapes, total_dim). Used
    by both client and server to agree on how the flat D-length vector
    maps back onto per-tensor shapes."""
    state = model.state_dict()
    keys = list(state.keys())
    shapes = [tuple(v.shape) for v in state.values()]
    total_dim = int(sum(int(np.prod(s)) for s in shapes))
    return keys, shapes, total_dim


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
        flat = np.concatenate([d.reshape(-1) for d in deltas])

        # ---- Public, server-broadcast Top-K support for this round ----
        # S is the same for every client (see module docstring), so this
        # is just a gather, not a client-side selection decision.
        support = np.frombuffer(fit_config["support_indices"], dtype=np.int64)
        sparse_vals = flat[support]  # length K, K << D

        # ---- SecAgg masking, restricted to the K support entries ----
        mask_start = time.time()
        combined_mask = np.zeros_like(sparse_vals)
        for other_idx, secret in self.pairwise_secrets.items():
            m = generate_mask(secret, round_num, sparse_vals.size, MASK_SCALE)
            combined_mask += m if self.client_idx < other_idx else -m
        masked_vals = (sparse_vals + combined_mask).astype(np.float32)
        mask_time = time.time() - mask_start

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "mask_time_sec": mask_time,
            "support_size": int(support.size),
        }
        # Single flat array of length K -- this (not D) is what actually
        # crosses the network, which is where the communication savings
        # over plain secagg.py come from.
        return [masked_vals], len(self.train_loader.dataset), metrics

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
    def __init__(self, test_loader, param_keys, param_shapes, total_dim,
                 top_k_fraction=TOP_K_FRACTION, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0

        self.param_keys = param_keys
        self.param_shapes = param_shapes
        self.total_dim = total_dim
        self.k = max(1, int(top_k_fraction * total_dim))

        # Round 1: no aggregate history exists yet to rank by magnitude,
        # so bootstrap with a fixed, publicly-seeded random K-subset.
        # Every client derives the identical set independently -- no
        # extra communication needed, it's just a shared constant.
        bootstrap_rng = np.random.RandomState(123)
        self.current_support = np.sort(
            bootstrap_rng.choice(total_dim, size=self.k, replace=False)
        ).astype(np.int64)

        print(f"Top-K SecAgg: D={total_dim} total params, K={self.k} "
              f"({top_k_fraction*100:.1f}% kept per round)")

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        support_bytes = self.current_support.tobytes()
        for _, fit_ins in fit_ins_list:
            fit_ins.config["server_round"] = server_round
            fit_ins.config["support_indices"] = support_bytes
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            return None, {}

        n_clients = len(results)
        # UNWEIGHTED sum of masked K-length vectors -- required for exact
        # mask cancellation, same reasoning as secagg.py. Every client
        # masked the SAME K coordinates this round, so cancellation is
        # exact here too, just over a shorter vector.
        summed = np.zeros(self.k, dtype=np.float64)
        mask_times = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            summed += arrays[0].astype(np.float64)
            mask_times.append(fit_res.metrics.get("mask_time_sec", 0.0))
        avg_sparse_delta = (summed / n_clients).astype(np.float32)

        # Scatter the K averaged values back into a full-size dense
        # delta; every non-selected coordinate is implicitly zero this
        # round (see "NEW LIMITATION" in the module docstring).
        full_delta = np.zeros(self.total_dim, dtype=np.float32)
        full_delta[self.current_support] = avg_sparse_delta

        global_state = self.global_model.state_dict()
        new_state = {}
        offset = 0
        for key, shape in zip(self.param_keys, self.param_shapes):
            n = int(np.prod(shape))
            delta_tensor = torch.tensor(
                full_delta[offset:offset + n].reshape(shape),
                dtype=global_state[key].dtype, device=global_state[key].device,
            )
            new_state[key] = global_state[key] + delta_tensor
            offset += n

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_mask_time = float(np.mean(mask_times)) if mask_times else 0.0
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | "
              f"K={self.k} ({self.k/self.total_dim*100:.1f}% of D) | "
              f"Avg client-side mask time: {avg_mask_time*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_secagg_topk.pth")

        # Pick NEXT round's support: top-K by |magnitude| of THIS round's
        # aggregate. This is public information (everyone already sees
        # the new global model), so choosing it costs no extra privacy.
        next_support = np.argpartition(np.abs(full_delta), -self.k)[-self.k:]
        self.current_support = np.sort(next_support).astype(np.int64)

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)

        return aggregated_params, {
            "accuracy": acc,
            "avg_mask_time_ms": avg_mask_time * 1000,
            "k": self.k,
            "sparsity_pct": self.k / self.total_dim * 100,
        }

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
        plt.title("Final Confusion Matrix (Secure Aggregation + Top-K Sparsification)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_secagg_topk.png")
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

    # Determine the flat parameter layout once, from a template model, so
    # client and server agree on how indices map onto tensors.
    template_model = IMUTransformerEncoder(config).to(DEVICE)
    param_keys, param_shapes, total_dim = get_param_layout(template_model)
    del template_model

    def client_fn(context):
        if hasattr(context, "node_id"):
            cid = int(context.node_id)
        elif hasattr(context, "node_config") and "cid" in context.node_config:
            cid = int(context.node_config["cid"])
        else:
            cid = 0
        client_idx = cid % len(client_datasets)
        return IMUClient(client_datasets[client_idx], client_idx, all_secrets[client_idx]).to_client()

    strategy = SaveModelStrategy(
        test_loader=test_loader,
        param_keys=param_keys,
        param_shapes=param_shapes,
        total_dim=total_dim,
        top_k_fraction=TOP_K_FRACTION,
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