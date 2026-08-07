"""
Federated HAR training with Gaussian-mechanism Differential Privacy.

Replaces the SAPM (Fisher top-k + quantization + permutation) pipeline.
Model, dataset, and overall FL loop are unchanged. What changed:

  - Quantization and permutation are REMOVED. They added compute and
    (per the SAPM docstring) didn't reduce bytes sent anyway, and neither
    one gave a provable privacy guarantee -- permutation is invertible by
    the server (it needs to be, to aggregate), and unbounded stochastic
    rounding has no calibrated sensitivity/epsilon behind it.

  - In their place: per-client delta clipping to a fixed L2 norm C,
    followed by a SINGLE calibrated Gaussian noise draw added to the
    SUM of clipped deltas at the server (not per-client), before
    averaging. This is standard DP-FedAvg (McMahan et al., 2017).

  - Cumulative privacy loss is tracked with Opacus's RDP accountant
    across all NUM_ROUNDS releases and printed every round and at the
    end, so you always know exactly how much epsilon has been spent.

WHY NOISE IS ADDED ONCE, TO THE SUM, NOT PER-CLIENT:
  If every client independently added noise before sending, the noise
  variances would ADD across clients. Instead: sum the CLIPPED (not yet
  noised) deltas, add exactly one Gaussian draw scaled to sigma*C to
  that sum, THEN divide by n_clients. The real signal in the sum scales
  with n_clients; the single noise draw does not -- so the averaged
  update's noise shrinks roughly as (sigma*C / n_clients). This is the
  main lever for keeping DP noise from swamping the model.

CAVEAT WORTH KNOWING BEFORE YOU TUNE THIS:
  With NUM_CLIENTS=3, that 1/n_clients dilution is weak -- most DP-FL
  work assumes hundreds-to-thousands of clients for exactly this reason.
  If accuracy collapses after enabling DP, the first things to check are
  (in order): is DP_TARGET_EPSILON too tight for 40 rounds x 3 clients,
  is DP_CLIP_NORM miscalibrated (see the pre_clip_l2_norm diagnostic
  printed every round -- it tells you the actual scale of client
  updates so you can set C sensibly instead of guessing), and only then
  whether you need to reduce what gets noised (e.g. apply DP only to a
  subset of layers, or bring back Fisher-based top-k as a pure
  dimensionality-reduction step -- NOT a privacy step this time).

Install once: pip install opacus --break-system-packages
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

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# ====================== PRIVACY (DP) HELPERS ======================
class dpu:
    @staticmethod
    def flatten(arrays):
        """Flatten a list of per-tensor numpy arrays into one 1D vector,
        remembering shapes/sizes so it can be split back apart later."""
        shapes = [a.shape for a in arrays]
        sizes = [a.size for a in arrays]
        flat = np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)
        return flat, shapes, sizes

    @staticmethod
    def unflatten(flat, shapes, sizes):
        out = []
        idx = 0
        for shape, size in zip(shapes, sizes):
            out.append(flat[idx: idx + size].reshape(shape))
            idx += size
        return out

    @staticmethod
    def clip_by_l2_norm(flat: np.ndarray, clip_norm: float):
        """Clip a flattened update to max L2 norm `clip_norm`. Returns the
        (possibly rescaled) vector AND the pre-clip norm, so callers can
        check whether clip_norm is well-calibrated to real update sizes."""
        norm = float(np.linalg.norm(flat))
        if norm > clip_norm and norm > 0:
            flat = flat * (clip_norm / norm)
        return flat, norm


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

# ---- differential privacy knobs ----
USE_PRIVACY = True
DP_CLIP_NORM = 1.0          # C: max L2 norm of ONE client's delta, taken
                             # over ALL parameters concatenated. This is
                             # the sensitivity the accountant relies on.
                             # DON'T guess this blind -- run one round
                             # with USE_PRIVACY=False first (or just read
                             # the printed pre_clip_l2_norm on round 1)
                             # to see what real update norms look like,
                             # then set C near the typical/median value.
                             # Too small -> clipping destroys signal.
                             # Too large -> noise (which scales with C)
                             # swamps signal. There's no free lunch here.
DP_TARGET_EPSILON = 8.0     # total privacy budget for the WHOLE run
DP_TARGET_DELTA = 1e-5      # should be << 1 / (largest client's dataset size)

# All NUM_CLIENTS participate every round (no random subsampling), so we
# tell the accountant sample_rate=1.0 -- there's no privacy amplification
# to claim here. If you later subsample clients per round, set this to
# the real participation fraction; the required noise for the same
# epsilon will drop.
DP_NOISE_MULTIPLIER = get_noise_multiplier(
    target_epsilon=DP_TARGET_EPSILON,
    target_delta=DP_TARGET_DELTA,
    sample_rate=1.0,
    steps=NUM_ROUNDS,
    accountant="rdp",
)

print(f"Using device: {DEVICE}")
print(f"Privacy: Gaussian-mechanism DP | enabled={USE_PRIVACY} | "
      f"clip_norm={DP_CLIP_NORM} | target_epsilon={DP_TARGET_EPSILON} | "
      f"target_delta={DP_TARGET_DELTA}")
print(f"[DP] Solved noise_multiplier sigma={DP_NOISE_MULTIPLIER:.4f} for "
      f"{NUM_ROUNDS} rounds at full ({NUM_CLIENTS}/{NUM_CLIENTS}) participation each round.")


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

    print("=" * 60)
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

    def fit(self, parameters, fit_config):
        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        use_privacy = fit_config.get("use_privacy", USE_PRIVACY)
        clip_norm = fit_config.get("dp_clip_norm", DP_CLIP_NORM)

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
        comm_no_privacy_bytes = sum(d.nbytes for d in deltas)

        if use_privacy:
            flat, shapes, sizes = dpu.flatten(deltas)
            flat, pre_clip_norm = dpu.clip_by_l2_norm(flat, clip_norm)
            out_arrays = [a.astype(np.float32) for a in dpu.unflatten(flat, shapes, sizes)]
            was_clipped = pre_clip_norm > clip_norm
        else:
            out_arrays = deltas
            full_flat = np.concatenate([d.reshape(-1) for d in deltas])
            pre_clip_norm = float(np.linalg.norm(full_flat))
            was_clipped = False

        comm_dense_bytes = sum(a.nbytes for a in out_arrays)

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "pre_clip_l2_norm": pre_clip_norm,   # sanity-check DP_CLIP_NORM against this
            "was_clipped": float(was_clipped),
            "comm_dense_bytes": comm_dense_bytes,
            "comm_no_privacy_bytes": comm_no_privacy_bytes,
        }
        return out_arrays, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, eval_config):
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
    def __init__(self, test_loader, use_privacy=USE_PRIVACY,
                 dp_clip_norm=DP_CLIP_NORM,
                 dp_noise_multiplier=DP_NOISE_MULTIPLIER,
                 dp_target_delta=DP_TARGET_DELTA, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0

        self.use_privacy = use_privacy
        self.privacy_clip_norm = dp_clip_norm
        self.privacy_noise_multiplier = dp_noise_multiplier
        self.privacy_target_delta = dp_target_delta
        self.privacy_accountant = RDPAccountant()

        self.total_comm_dense_bytes = 0
        self.total_comm_no_privacy_bytes = 0

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_privacy"] = self.use_privacy
            fit_ins.config["dp_clip_norm"] = self.privacy_clip_norm
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        shapes = [tuple(global_state[k].shape) for k in keys]
        sizes = [global_state[k].numel() for k in keys]

        summed_flat = None
        n_clients = 0
        round_comm_dense_bytes = 0
        round_comm_no_privacy_bytes = 0
        pre_clip_norms = []
        clip_count = 0

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            flat, _, _ = dpu.flatten(arrays)
            summed_flat = flat if summed_flat is None else summed_flat + flat
            n_clients += 1

            round_comm_dense_bytes += fit_res.metrics.get("comm_dense_bytes", 0)
            round_comm_no_privacy_bytes += fit_res.metrics.get("comm_no_privacy_bytes", 0)
            pre_clip_norms.append(fit_res.metrics.get("pre_clip_l2_norm", 0.0))
            clip_count += int(fit_res.metrics.get("was_clipped", 0.0))

        if self.use_privacy:
            # Gaussian mechanism, sensitivity = clip_norm (one client's
            # bounded contribution), applied ONCE to the SUM -- see
            # module docstring for why this beats per-client noising.
            noise = np.random.normal(
                loc=0.0,
                scale=self.privacy_noise_multiplier * self.privacy_clip_norm,
                size=summed_flat.shape,
            )
            noised_sum = summed_flat + noise
            self.privacy_accountant.step(
                noise_multiplier=self.privacy_noise_multiplier, sample_rate=1.0
            )
        else:
            noised_sum = summed_flat

        avg_flat = noised_sum / max(1, n_clients)
        avg_deltas = dpu.unflatten(avg_flat, shapes, sizes)

        new_state = {}
        for k, delta in zip(keys, avg_deltas):
            new_state[k] = global_state[k] + torch.tensor(
                delta, dtype=global_state[k].dtype, device=global_state[k].device
            )

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_pre_clip_norm = float(np.mean(pre_clip_norms)) if pre_clip_norms else 0.0

        self.total_comm_dense_bytes += round_comm_dense_bytes
        self.total_comm_no_privacy_bytes += round_comm_no_privacy_bytes

        eps_so_far = (
            self.privacy_accountant.get_epsilon(self.privacy_target_delta)
            if self.use_privacy else 0.0
        )

        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f}")
        print(f"  [dp] avg pre-clip L2 norm: {avg_pre_clip_norm:.4f} "
              f"(clip_norm={self.privacy_clip_norm}, {clip_count}/{n_clients} clients clipped this round) | "
              f"cumulative epsilon: {eps_so_far:.3f} / budget {DP_TARGET_EPSILON}")
        print(f"  [comm] {round_comm_dense_bytes/1e6:.3f} MB sent "
              f"(no-privacy baseline: {round_comm_no_privacy_bytes/1e6:.3f} MB)")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), "best_model.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            final_eps = (
                self.privacy_accountant.get_epsilon(self.privacy_target_delta)
                if self.use_privacy else 0.0
            )
            print(f"\n[DP] FINAL privacy spent: epsilon={final_eps:.4f} at delta={self.privacy_target_delta} "
                  f"(target was epsilon={DP_TARGET_EPSILON})")
            print(f"[comm] Total sent: {self.total_comm_dense_bytes/1e6:.2f} MB "
                  f"(no-privacy baseline: {self.total_comm_no_privacy_bytes/1e6:.2f} MB)")

        return aggregated_params, {"accuracy": acc}

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
        plt.title("Final Confusion Matrix (DP)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("final_confusion_matrix_dp.png")
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

    strategy = SaveModelStrategy(
        test_loader=test_loader,
        use_privacy=USE_PRIVACY,
        dp_clip_norm=DP_CLIP_NORM,
        dp_noise_multiplier=DP_NOISE_MULTIPLIER,
        dp_target_delta=DP_TARGET_DELTA,
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