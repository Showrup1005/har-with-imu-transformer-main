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
    model in configure_fit, so it costs no extra round trip). Since every
    client masks and sends values at the exact same K indices, the
    original pairwise-mask cancellation proof from secagg.py applies
    completely unchanged -- just with vectors of length K instead of
    length D.

    S is chosen by an EXPLOIT / EXPLORE split of the K budget each round:
        - EXPLOIT (~70% of K): the coordinates with the largest observed
          |delta| out of everything measured so far, from a persistent
          magnitude_estimate[] array that is updated ONLY at coordinates
          actually included in a round's support, and left untouched
          everywhere else.
        - EXPLORE (~30% of K): coordinates drawn from a fixed, publicly
          seeded random visiting order that cycles through ALL D
          coordinates over time (a shared pointer, advanced every
          round), guaranteeing every parameter gets directly measured at
          least once roughly every D / explore_k rounds.
    An earlier version of this file picked next-round's S as "top-K by
    |magnitude| of this round's aggregated delta" -- but that delta is
    zero everywhere except at the current S (nothing else was ever
    communicated), so "top-K of a mostly-zero vector" just re-selects
    the same S forever. In a real run that froze training onto whatever
    random 10% of parameters the round-1 bootstrap happened to pick,
    and accuracy never moved off a near-uniform-guess baseline. The
    exploit/explore split above is the fix: magnitude_estimate is never
    implicitly zeroed for coordinates outside S, and the explore slice
    guarantees the whole parameter space gets sampled over the run
    rather than the support permanently collapsing.

    This is a real accuracy/communication trade-off relative to the
    papers' approach: letting each client keep its OWN locally-optimal
    top-K (rather than a shared, exploit/explore-driven one) would track
    each client's individual large-gradient coordinates more precisely
    and without the discovery lag inherent in round-robin exploration.
    What we keep is the thing that matters most for a demo: exact mask
    cancellation with zero extra cryptographic machinery, genuine
    communication savings that scale with K/D, and (with the fix above)
    guaranteed full-parameter-space coverage over the course of
    training.

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
      that round. The explore mechanism bounds how stale this can get:
      every coordinate is guaranteed to be directly measured at least
      once every D / explore_k rounds (with the defaults below, roughly
      every ~33 rounds for D≈1.5M), so a coordinate that starts mattering
      will eventually be discovered and, once it shows a large delta,
      pulled into the exploit set for continued refinement. With a small
      K or a small EXPLORE_FRACTION_OF_K this discovery lag grows; those
      two knobs are the trade-off against per-round communication
      savings.
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
EXPLORE_FRACTION_OF_K = 0.30  # fraction of each round's K budget spent
                          # on round-robin exploration of coordinates
                          # not yet directly measured, vs. exploiting
                          # the best coordinates found so far. Higher =
                          # faster full-space coverage, less budget
                          # spent refining known-important coordinates.

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
        self.explore_k = max(1, int(EXPLORE_FRACTION_OF_K * self.k))
        self.exploit_k = self.k - self.explore_k

        # Persistent, server-side-only bookkeeping (never communicated
        # as-is -- only the resulting support S is sent to clients).
        # magnitude_estimate is updated ONLY at coordinates we actually
        # measure in a given round; everywhere else keeps its last known
        # value rather than being implicitly treated as zero. This is
        # what prevents the support from collapsing onto a fixed subset
        # (see module docstring for the bug this fixes).
        self.magnitude_estimate = np.zeros(total_dim, dtype=np.float64)
        self.known_mask = np.zeros(total_dim, dtype=bool)

        # Fixed, publicly-seeded random visiting order used for
        # round-robin exploration -- guarantees every one of the D
        # coordinates gets directly measured over the course of
        # training, regardless of where it sits in the flattened vector
        # (a raw index-order sweep would spend many rounds stuck inside
        # just the first tensor or two).
        order_rng = np.random.RandomState(2024)
        self.explore_order = order_rng.permutation(total_dim).astype(np.int64)
        self.explore_ptr = 0

        # Round 1: no observations exist yet, so spend the entire K
        # budget as an initial exploration batch rather than splitting
        # into an exploit set that has nothing informative to exploit.
        self.current_support = np.sort(self.explore_order[:self.k]).astype(np.int64)
        self.explore_ptr = self.k % total_dim

        print(f"Top-K SecAgg: D={total_dim} total params, K={self.k} "
              f"({top_k_fraction*100:.1f}% kept per round) | "
              f"exploit={self.exploit_k}, explore={self.explore_k} "
              f"(full coverage in ~{-(-total_dim // self.explore_k)} rounds)")

    def _next_explore_batch(self, n, exclude_set):
        """Pull n coordinates from the fixed round-robin visiting order,
        skipping any already claimed by the exploit set this round."""
        picked = []
        scanned = 0
        while len(picked) < n and scanned < self.total_dim:
            cand = int(self.explore_order[self.explore_ptr])
            self.explore_ptr = (self.explore_ptr + 1) % self.total_dim
            scanned += 1
            if cand not in exclude_set:
                picked.append(cand)
        return picked

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

        # ---- Update magnitude knowledge, then pick NEXT round's support ----
        # Only touch the estimate at coordinates we actually measured this
        # round; everything else keeps its previous value untouched.
        self.magnitude_estimate[self.current_support] = np.abs(full_delta[self.current_support])
        self.known_mask[self.current_support] = True

        # EXPLOIT: the exploit_k coordinates with the largest measured
        # |delta| among everything observed so far.
        observed_idx = np.flatnonzero(self.known_mask)
        if observed_idx.size <= self.exploit_k:
            exploit = observed_idx
        else:
            top_within_observed = np.argpartition(
                self.magnitude_estimate[observed_idx], -self.exploit_k
            )[-self.exploit_k:]
            exploit = observed_idx[top_within_observed]
        exploit_set = set(int(i) for i in exploit)

        # EXPLORE: round-robin through the fixed visiting order to fill
        # the rest of the budget with coordinates not already exploited,
        # guaranteeing eventual full coverage of the parameter space.
        explore = self._next_explore_batch(self.k - len(exploit_set), exploit_set)

        self.current_support = np.array(
            sorted(exploit_set) + sorted(explore), dtype=np.int64
        )
        coverage_pct = observed_idx.size / self.total_dim * 100
        print(f"  -> next support: exploit={len(exploit_set)}, explore={len(explore)} | "
              f"cumulative coverage so far: {coverage_pct:.1f}% of all params")

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