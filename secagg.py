"""
Federated HAR training with Secure Aggregation (SecAgg) + Round-Synchronized
K-Sparsification.

THE PROBLEM THIS FILE ADDS ON TOP OF secagg.py:
    Plain SecAgg (secagg.py) sends a masked vector the same size as the
    FULL model every round. That's fine for a tiny model but becomes the
    communication bottleneck for large ones. Sparsification fixes this by
    having each client send only K of its D update entries instead of
    all of them.

    The catch (this is the exact issue raised in "Secure Aggregation with
    Top-K Sparsification in Decentralized Federated Learning", Tang, Zhu
    & Tang 2026, and independently in "Secure Aggregation Meets
    Sparsification in Decentralized Learning" (CESAR), 2024): if each
    client picks its OWN top-K indices by local gradient/Fisher-score
    magnitude, different clients keep different index subsets. SecAgg's
    pairwise +m/-m cancellation only works when both sides of a pair mask
    the SAME coordinates -- mismatched supports leave unmatched mask
    entries that don't cancel and corrupt the sum. Those papers solve
    this with real cryptographic engineering (secret-sharing over index
    sets, coded computing, etc.) to reconcile mismatched supports.

THE SIMPLIFICATION USED HERE, STATED EXPLICITLY:
    All clients sparsify to the SAME K-index subset each round, derived
    INDEPENDENTLY by every client and the server from the public round
    number via a shared PRNG (`support_for_round` below) -- no index list
    is ever transmitted, so the K-value communication savings are real
    (not K values plus a separate K-index list). Since every client masks
    and sends values at the exact same K indices, the original pairwise-
    mask cancellation proof from secagg.py applies completely unchanged,
    just over length-K vectors instead of length-D.

    Calling this "Top-K" would overstate it: the support is chosen purely
    from the round number, unrelated to which coordinates actually have
    the largest gradient magnitude, so this is really RANDOM-K
    sparsification, not magnitude-based Top-K. That's the honest
    trade-off against the papers' approach -- their per-client
    magnitude-informed supports converge faster per communicated bit, at
    the cost of the index-reconciliation machinery described above. What
    we keep here is a demo that's simple, has zero persistent
    support-selection state (so it cannot get stuck re-selecting the same
    subset -- see BUG #1 below), and needs no extra round trip or index
    transmission.

THREE BUGS FOUND AND FIXED DURING DEVELOPMENT OF THIS FILE (kept here for
a record of what broke and why, since each produced the exact same
symptom -- accuracy frozen at a near-uniform-guess baseline for the whole
run):

    BUG #1 -- naive support re-selection collapsed to a fixed subset.
    An early version picked next-round's support as "top-K by |magnitude|
    of this round's aggregated delta." That delta is zero everywhere
    except the current support (nothing else was ever communicated), so
    "top-K of a mostly-zero vector" just re-selects the same support
    forever -- training got permanently stuck retraining whatever random
    10% of parameters the round-1 bootstrap happened to pick. Fixed (for
    a while) with an exploit/explore scheme, and now fixed more simply by
    switching to the round-number-PRNG support above, which has no
    persistent state to collapse.

    BUG #2 -- the server's global model and clients' round-1 starting
    point were different, unrelated random initializations. The strategy
    built its own `self.global_model` with an independent random init,
    while Flower's default `initialize_parameters` behavior separately
    queried one client's own random init to broadcast for round 1.
    Clients trained from -- and computed deltas relative to -- that
    client's init, but `aggregate_fit` added those deltas onto the
    strategy's unrelated init instead, producing an internally incoherent
    model (mismatched attention projections, LayerNorm gamma/beta pairs,
    etc.) whose output collapses toward a near-constant prediction. Fixed
    by `SaveModelStrategy.initialize_parameters`, which hands out the
    strategy's own `global_model` weights as the round-1 broadcast.

    BUG #3 -- client SecAgg identity assigned via `node_id % NUM_CLIENTS`
    could collide. SecAgg's pairwise-mask cancellation requires the
    NUM_CLIENTS simultaneously-participating clients to map to NUM_CLIENTS
    distinct roles (0, 1, 2, ...) every single round, with zero
    collisions -- but Flower's `node_id` values aren't guaranteed to
    distribute cleanly mod NUM_CLIENTS. If two clients ever collided onto
    the same role, they'd derive the same pairwise-secret bundle and the
    same data partition, one role would go entirely unrepresented for
    that round, and the +m/-m cancellation invariant would break --
    injecting raw MASK_SCALE-magnitude noise directly into the aggregate.
    Fixed by having the SERVER assign roles explicitly and deterministically
    from `ClientProxy.cid` (Flower-guaranteed unique and stable for the
    life of the simulation), communicated to each client via `fit_config`
    -- no reliance on `node_id` arithmetic at all. See
    `SaveModelStrategy.configure_fit` / `configure_evaluate`.

WHAT DOESN'T CHANGE FROM secagg.py:
    - The ECDH key exchange, HKDF derivation, and per-round PRG masking
      math are identical (see generate_pairwise_secrets / generate_mask
      below) -- only the SIZE of the vector being masked changes, from D
      (full model) to K (support size).
    - The server still only ever sees a SUM of masked values, never an
      individual client's true update -- sparsification doesn't weaken
      that guarantee, it only changes how many coordinates are summed.
    - Same limitations apply: no dropout resilience, unweighted averaging
      only, numpy RandomState (not a CSPRNG) as the PRG, and a 3-client
      anonymity set far smaller than production deployments.

NEW LIMITATION FROM (RANDOM-K) SPARSIFICATION:
    - Non-selected coordinates are implicitly zero-delta for that round,
      and WHICH coordinates get selected is unrelated to their actual
      importance (unlike true magnitude-based Top-K). Over many rounds
      every coordinate gets roughly equal, evenly-distributed attention
      in expectation, but there's no mechanism prioritizing
      high-gradient parameters the way the papers' per-client-informed
      schemes do -- expect slower convergence per communicated bit than
      genuine Top-K, in exchange for the simplicity/robustness above.
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
                          # regardless of magnitude; see secagg.py). If
                          # your real per-round deltas are much smaller
                          # than this (check the "Delta L2 norm" diagnostic
                          # printed each round), consider lowering it --
                          # it's always safe for accuracy, since it
                          # cancels exactly either way, and a smaller
                          # value reduces float32 precision loss when the
                          # noise and signal are added together.
TOP_K_FRACTION = 0.10    # fraction of total model parameters kept each
                          # round (see support_for_round below).

print(f"Using device: {DEVICE}")
print(f"Privacy strategy: Secure Aggregation (X25519 ECDH pairwise masking) "
      f"+ Random-{TOP_K_FRACTION*100:.0f}% round-synchronized sparsification")


# ====================== KEY EXCHANGE (done once, before training) ======================
def generate_pairwise_secrets(num_clients: int):
    """Real ECDH key exchange between every pair of ROLES (0..num_clients-1).
    Identical to secagg.py -- unaffected by sparsification or role
    assignment, since it just establishes per-pair shared secrets, keyed
    by role index, not by any client-transport identifier."""
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
    the identical (shared_secret, round_num, size) triple -- since the
    support is public and derived identically on both sides, size is
    always identical too, so the resulting mask vectors still cancel
    exactly on summation."""
    seed = mask_seed_for_round(shared_secret, round_num)
    rng = np.random.RandomState(seed % (2 ** 31 - 1))
    return rng.normal(0.0, scale, size=size).astype(np.float32)


def support_for_round(round_num: int, total_dim: int, k: int) -> np.ndarray:
    """Purely public, round-number-derived support set. Both client and
    server compute this identically and independently -- no index list
    is ever transmitted, so the K-value communication savings are real.
    See module docstring: this makes the scheme RANDOM-K sparsification,
    not magnitude-based Top-K."""
    rng = np.random.RandomState(round_num % (2 ** 31 - 1))
    return np.sort(rng.choice(total_dim, size=k, replace=False)).astype(np.int64)


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
    """Deliberately holds NO fixed role/data-partition/secret-bundle at
    construction time -- Flower's `node_id` is not a safe basis for that
    (see BUG #3 in the module docstring). Instead, every fit()/evaluate()
    call resolves its role fresh from the server-provided `client_role`
    in the round's config, and looks up the corresponding data partition
    and pairwise-secret bundle from the full lists/dicts held here."""

    def __init__(self, all_client_datasets, all_secrets):
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.all_client_datasets = all_client_datasets  # list[Subset], indexed by role
        self.all_secrets = all_secrets                  # {role: {other_role: shared_secret_bytes}}

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
        role = fit_config["client_role"]
        pairwise_secrets = self.all_secrets[role]
        train_loader = DataLoader(
            self.all_client_datasets[role], batch_size=config["batch_size"], shuffle=True, num_workers=0
        )

        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        round_num = fit_config.get("server_round", 0)

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for _ in range(LOCAL_EPOCHS):
            for batch in train_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()

                self.optimizer.zero_grad()
                output = self.model({"imu": imu})
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        new_state = self.model.state_dict()
        keys = list(new_state.keys())
        deltas = [(new_state[k] - old_state[k]).cpu().numpy().astype(np.float32) for k in keys]
        flat = np.concatenate([d.reshape(-1) for d in deltas])

        # ---- Round-synchronized support: computed locally, not received ----
        total_dim = flat.size
        k = max(1, int(TOP_K_FRACTION * total_dim))
        support = support_for_round(round_num, total_dim, k)
        sparse_vals = flat[support]  # length K, K << D

        # ---- SecAgg masking, restricted to the K support entries ----
        mask_start = time.time()
        combined_mask = np.zeros_like(sparse_vals)
        for other_role, secret in pairwise_secrets.items():
            m = generate_mask(secret, round_num, sparse_vals.size, MASK_SCALE)
            combined_mask += m if role < other_role else -m
        masked_vals = (sparse_vals + combined_mask).astype(np.float32)
        mask_time = time.time() - mask_start

        metrics = {
            "train_loss": total_loss / max(num_batches, 1),
            "mask_time_sec": mask_time,
            "support_size": int(support.size),
            "client_role": role,
        }
        # Single flat array of length K -- this (not D) is what actually
        # crosses the network, which is where the communication savings
        # over plain secagg.py come from.
        return [masked_vals], len(train_loader.dataset), metrics

    def evaluate(self, parameters, eval_config):
        role = eval_config.get("client_role", 0)
        eval_loader = DataLoader(
            self.all_client_datasets[role], batch_size=config["batch_size"], shuffle=False, num_workers=0
        )
        self.set_parameters(parameters)
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                imu = batch["imu"].to(DEVICE).float()
                label = batch["label"].to(DEVICE).long()
                output = self.model({"imu": imu})
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        return float(0.0), len(eval_loader.dataset), {"accuracy": accuracy}


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

        # Lazily resolved on the first configure_fit/configure_evaluate
        # call, once client_manager actually knows about the clients:
        # {ClientProxy.cid (str): role (int, 0..NUM_CLIENTS-1)}. Built
        # ONCE from a sorted order of cids -- guaranteed unique and
        # stable for the life of the simulation -- so every round maps
        # the same physical client to the same role with zero collisions
        # (see BUG #3 in the module docstring).
        self.cid_to_role = None

        print(f"Round-synchronized K-sparsified SecAgg: D={total_dim} total params, "
              f"K={self.k} ({top_k_fraction*100:.1f}% kept per round, RANDOM-K -- "
              f"see module docstring)")

    def _ensure_roles(self, proxy_ins_pairs):
        if self.cid_to_role is None:
            sorted_cids = sorted(cp.cid for cp, _ in proxy_ins_pairs)
            assert len(sorted_cids) == len(set(sorted_cids)), (
                "Duplicate ClientProxy.cid values seen in the same round -- "
                "this should be impossible (cid is Flower's unique client "
                "identifier), but if it ever happens the role assignment "
                "below would silently collide, reproducing BUG #3."
            )
            self.cid_to_role = {cid: i for i, cid in enumerate(sorted_cids)}
            assert len(set(self.cid_to_role.values())) == len(self.cid_to_role), (
                "Role assignment produced duplicate roles -- this must never "
                "happen; aborting rather than silently corrupting SecAgg "
                "mask cancellation."
            )
            print(f"Assigned stable SecAgg roles via sorted ClientProxy.cid: "
                  f"{self.cid_to_role}")

    def initialize_parameters(self, client_manager):
        """Hand out THIS strategy's own global_model weights as the
        round-1 broadcast, instead of letting Flower's default behavior
        query a client for its own separately-initialized random model
        (see BUG #2 in the module docstring)."""
        arrays = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        return ndarrays_to_parameters(arrays)

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        self._ensure_roles(fit_ins_list)
        for client_proxy, fit_ins in fit_ins_list:
            fit_ins.config["server_round"] = server_round
            fit_ins.config["client_role"] = self.cid_to_role[client_proxy.cid]
        return fit_ins_list

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_ins_list = super().configure_evaluate(server_round, parameters, client_manager)
        self._ensure_roles(eval_ins_list)
        for client_proxy, eval_ins in eval_ins_list:
            eval_ins.config["client_role"] = self.cid_to_role[client_proxy.cid]
        return eval_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            return None, {}

        # Same support every client used this round, recomputed locally
        # (never received) from the public round number.
        support = support_for_round(server_round, self.total_dim, self.k)

        n_clients = len(results)
        # UNWEIGHTED sum of masked K-length vectors -- required for exact
        # mask cancellation, same reasoning as secagg.py. Every client
        # masked the SAME K coordinates this round, so cancellation is
        # exact here too, just over a shorter vector.
        summed = np.zeros(self.k, dtype=np.float64)
        mask_times = []
        train_losses = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            summed += arrays[0].astype(np.float64)
            mask_times.append(fit_res.metrics.get("mask_time_sec", 0.0))
            train_losses.append(fit_res.metrics.get("train_loss", float("nan")))
        avg_sparse_delta = (summed / n_clients).astype(np.float32)

        # DIAGNOSTIC: L2 norm of the recovered (unmasked, averaged) true
        # delta at this round's support. If this is ~0 every round, the
        # freeze is upstream of aggregation (clients aren't producing
        # real updates). If it's clearly nonzero but accuracy still never
        # moves, the freeze is downstream of aggregation.
        delta_norm = float(np.linalg.norm(avg_sparse_delta))
        avg_train_loss = float(np.nanmean(train_losses)) if train_losses else float("nan")

        # ---- Anomaly guards ----
        # A single UNCANCELLED pairwise mask (e.g. from a role collision --
        # see BUG #3) has expected L2 norm ~= MASK_SCALE * sqrt(K) even
        # after averaging by n_clients (since averaging only shrinks it by
        # 1/n_clients, not away entirely). Real per-parameter deltas are
        # typically orders of magnitude smaller than MASK_SCALE. If we see
        # a delta anywhere near that residual-noise scale, warn loudly
        # BEFORE applying it -- this is the exact signature that produced
        # a permanently NaN-collapsed model in earlier runs (round 1 had a
        # real-looking but huge delta norm; every round after was NaN).
        expected_uncancelled_residual = MASK_SCALE * np.sqrt(self.k) / n_clients
        if delta_norm > 0.1 * expected_uncancelled_residual:
            print(f"  [WARNING] Delta L2 norm ({delta_norm:.3e}) is within 10x of the "
                  f"expected magnitude of a single UNCANCELLED SecAgg mask "
                  f"({expected_uncancelled_residual:.3e}). This is the signature of "
                  f"mask cancellation failing (e.g. two clients assigned the same "
                  f"SecAgg role this round) rather than a real gradient update -- "
                  f"check the role map printed at round 1 for duplicate roles.")
        if not np.isfinite(delta_norm) or not np.isfinite(avg_train_loss):
            print(f"  [WARNING] Non-finite value detected (delta_norm={delta_norm}, "
                  f"avg_train_loss={avg_train_loss}). The model weights from a PRIOR "
                  f"round likely already contain NaN/Inf (e.g. from an earlier "
                  f"cancellation failure blowing up a weight update) -- once that "
                  f"happens the forward pass produces NaN forever after, which is "
                  f"consistent with the model then collapsing to a single constant "
                  f"predicted class every round.")

        # Scatter the K averaged values back into a full-size dense
        # delta; every non-selected coordinate is implicitly zero this
        # round (see "NEW LIMITATION" in the module docstring).
        full_delta = np.zeros(self.total_dim, dtype=np.float32)
        full_delta[support] = avg_sparse_delta

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
              f"Avg train_loss: {avg_train_loss:.6f} | "
              f"Delta L2 norm: {delta_norm:.6e} | "
              f"Avg client-side mask time: {avg_mask_time*1000:.2f}ms")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model_secagg_topk.pth")

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
            # DIAGNOSTIC: if the model has collapsed to (near-)always
            # predicting one class, num_distinct_preds will be 1 (or very
            # low) and max_class_share will be high -- a strong sign the
            # freeze is in the model itself (dead/saturated units, or a
            # learning-rate/init problem) rather than in the FL/SecAgg
            # plumbing above it.
            unique, counts = np.unique(all_preds, return_counts=True)
            max_share = counts.max() / all_preds.size
            print(f"  [diag] distinct predicted classes: {unique.size} | "
                  f"largest single-class share: {max_share*100:.1f}%")
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
        plt.title("Final Confusion Matrix (Secure Aggregation + Random-K Sparsification)")
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

    print("Performing real X25519 ECDH key exchange between all client roles...")
    all_secrets = generate_pairwise_secrets(NUM_CLIENTS)
    print(f"Derived {sum(len(v) for v in all_secrets.values())} directed pairwise secrets "
          f"({NUM_CLIENTS * (NUM_CLIENTS - 1) // 2} unordered pairs) via ECDH.\n")

    # Determine the flat parameter layout once, from a template model, so
    # client and server agree on how indices map onto tensors.
    template_model = IMUTransformerEncoder(config).to(DEVICE)
    param_keys, param_shapes, total_dim = get_param_layout(template_model)
    del template_model

    def client_fn(context):
        # No node_id-based role logic here at all (see BUG #3): every
        # client gets the FULL set of datasets/secrets and resolves its
        # actual role fresh, each round, from the server-provided
        # `client_role` in fit_config/eval_config.
        return IMUClient(client_datasets, all_secrets).to_client()

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