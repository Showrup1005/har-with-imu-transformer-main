"""
Federated HAR training with standard Differential Privacy (DP-FedAvg).

This is the McMahan et al. (2017) "Learning Differentially Private
Recurrent Language Models" client-level DP mechanism, adapted here to
per-round updates:

    1. Each client computes its local update (delta = new_weights - old_weights)
       after local training, exactly as in the non-private baseline.
    2. The ENTIRE update (flattened across all layers into one vector) is
       L2-clipped to a maximum norm C: if ||delta|| > C, scale it down to
       have norm exactly C. This bounds the maximum influence any single
       client's update can have on the aggregate.
    3. Independent Gaussian noise N(0, (sigma * C)^2) is added to every
       coordinate of the clipped update, where sigma is the noise
       multiplier. This is the standard Gaussian mechanism for
       differential privacy.
    4. The full, dense, noised update is sent to the server -- nothing is
       sparsified, quantized, or permuted. Every coordinate is
       transmitted, just perturbed.

This gives a formal, quantifiable per-round privacy guarantee governed by
(C, sigma): larger sigma / smaller C => stronger privacy, more noise,
worse accuracy. Rigorously composing the guarantee across all NUM_ROUNDS
rounds into a single (epsilon, delta) requires an accountant (e.g. the
Opacus or Google dp-accounting RDP accountant) -- this script reports the
mechanism's raw parameters (clip norm, noise multiplier, effective noise
std) each round rather than a composed epsilon, since the exact number
depends on the accounting method and client sampling probability you
choose to report. Note this is a fundamentally different -- and heavier
-- privacy mechanism than SAPM: see the comparison notes at the bottom of
this file.
"""

import flwr as fl
import torch
import numpy as np
import json
import math
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
DP_DELTA = 1e-5  # standard choice: delta should be << 1/(num training examples)

# ---- DP strategy knobs ----
USE_DP = True
MAX_GRAD_NORM = 1.0     # C: L2 clipping norm applied to each client's full flattened update

# NOISE_MULTIPLIER (sigma) needs to be calibrated to model size: the total
# L2 norm of the noise vector scales as sqrt(num_params) * sigma * C. Our
# earlier run used the SMALLEST sigma that avoided numerical blow-up
# (sigma ~= 1/sqrt(num_params)) -- that is the WEAKEST defensible privacy
# level, not a "good privacy" one, and isn't a fair comparison against
# SAPM. PRIVACY_LEVEL below picks a multiple of that minimal baseline;
# "strong" is a reasonable default for a real privacy claim. Override
# PRIVACY_LEVEL or set NOISE_MULTIPLIER directly for a custom point.
PRIVACY_LEVEL = "moderate"   # one of: "minimal", "moderate", "strong", "very_strong"
_PRIVACY_LEVEL_MULTIPLIERS = {
    "minimal": 1,
    "moderate": 10,
    "strong": 30,
    "very_strong": 100,
}

_NUM_MODEL_PARAMS = sum(p.numel() for p in IMUTransformerEncoder(config).parameters())
_BASELINE_NOISE_MULTIPLIER = 1.0 / (_NUM_MODEL_PARAMS ** 0.5)
NOISE_MULTIPLIER = _BASELINE_NOISE_MULTIPLIER * _PRIVACY_LEVEL_MULTIPLIERS[PRIVACY_LEVEL]


def gaussian_mechanism_epsilon(noise_multiplier: float, delta: float) -> float:
    """Standard analytic (epsilon, delta)-DP bound for a SINGLE application
    of the Gaussian mechanism with sensitivity normalized to 1 (i.e.
    noise std = noise_multiplier * sensitivity). This is the textbook
    Dwork & Roth bound: sigma >= sqrt(2 ln(1.25/delta)) / epsilon, solved
    for epsilon. Valid for noise_multiplier roughly >= 1; for the small
    per-round multiplier used here this is an approximation of the
    per-round privacy cost, not the ONLY source of privacy loss --
    composing NUM_ROUNDS of these is what determines your total epsilon
    (see the note printed below)."""
    return math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier


print(f"Using device: {DEVICE}")
print(f"Model has {_NUM_MODEL_PARAMS:,} parameters | baseline (minimal) noise_multiplier = {_BASELINE_NOISE_MULTIPLIER:.6f}")
print(f"Privacy strategy: DP-FedAvg | enabled={USE_DP} | level={PRIVACY_LEVEL} "
      f"({_PRIVACY_LEVEL_MULTIPLIERS[PRIVACY_LEVEL]}x baseline) | clip_norm={MAX_GRAD_NORM} | "
      f"noise_multiplier={NOISE_MULTIPLIER:.6f}")

_eps_per_round = gaussian_mechanism_epsilon(NOISE_MULTIPLIER, DP_DELTA)
_eps_loose_total = _eps_per_round * NUM_ROUNDS  # basic (loose, linear) composition -- a conservative UPPER bound
print(f"Per-round Gaussian-mechanism epsilon (delta={DP_DELTA}): {_eps_per_round:.4f}")
print(f"Approx. TOTAL epsilon over {NUM_ROUNDS} rounds:")
print(f"  - basic (linear) composition, loose upper bound: {_eps_loose_total:.2f}")
if _eps_per_round > 50:
    # exp(epsilon) in the advanced-composition formula overflows once
    # epsilon is already this large -- and a per-round epsilon this big
    # means the noise is too weak to provide meaningful privacy at all,
    # so a tighter composed number wouldn't be a meaningful claim anyway.
    print("  - advanced composition: skipped (per-round epsilon is already so large that this "
          "noise level provides no meaningful privacy guarantee -- pick a stronger PRIVACY_LEVEL)")
else:
    _eps_advanced_total = (_eps_per_round * math.sqrt(2 * NUM_ROUNDS * math.log(1 / DP_DELTA))
                            + NUM_ROUNDS * _eps_per_round * (math.exp(_eps_per_round) - 1))
    print(f"  - advanced composition (Dwork et al.), tighter but still not optimal: {_eps_advanced_total:.2f}")
print("  NOTE: these are approximate, non-tight bounds for reporting purposes only. For a rigorous "
      "number to put in your thesis, use a proper RDP/moments accountant (e.g. the 'opacus' or "
      "'dp-accounting' Python packages), which will typically give a noticeably SMALLER (tighter, "
      "better) epsilon than the advanced-composition number above for the same noise level.")

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
        # Guard against a NaN/Inf global model (e.g. from an earlier round's
        # noise pushing weights to extreme values) so it can't silently
        # propagate forever -- replace any non-finite values with 0 and warn.
        cleaned = []
        found_bad = False
        for v in params:
            if not np.all(np.isfinite(v)):
                found_bad = True
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned.append(v)
        if found_bad:
            print("WARNING: received non-finite global parameters, sanitized to 0 for this client's local copy.")
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), cleaned)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, fit_config):
        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        use_dp = fit_config.get("use_dp", USE_DP)
        max_norm = fit_config.get("max_grad_norm", MAX_GRAD_NORM)
        noise_mult = fit_config.get("noise_multiplier", NOISE_MULTIPLIER)

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

        pre_clip_norm = float(np.sqrt(sum(np.sum(d.astype(np.float64) ** 2) for d in deltas)))

        if use_dp:
            # Step 1: clip the WHOLE update (across all layers together) to L2 norm max_norm.
            clip_factor = min(1.0, max_norm / (pre_clip_norm + 1e-12))
            deltas = [d * clip_factor for d in deltas]

            # Step 2: add iid Gaussian noise to every coordinate of the clipped update.
            noise_std = noise_mult * max_norm
            deltas = [d + np.random.normal(0.0, noise_std, size=d.shape).astype(np.float32) for d in deltas]

        post_norm = float(np.sqrt(sum(np.sum(d.astype(np.float64) ** 2) for d in deltas)))

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "pre_clip_norm": pre_clip_norm,
            "post_dp_norm": post_norm,
        }
        return deltas, len(self.train_loader.dataset), metrics

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
    def __init__(self, test_loader, use_dp=USE_DP, max_grad_norm=MAX_GRAD_NORM,
                 noise_multiplier=NOISE_MULTIPLIER, **kwargs):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.global_model = IMUTransformerEncoder(config).to(DEVICE)
        self.best_acc = 0.0
        self.use_dp = use_dp
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_dp"] = self.use_dp
            fit_ins.config["max_grad_norm"] = self.max_grad_norm
            fit_ins.config["noise_multiplier"] = self.noise_multiplier
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0
        pre_norms, post_norms = [], []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            pre_norms.append(fit_res.metrics.get("pre_clip_norm", 0.0))
            post_norms.append(fit_res.metrics.get("post_dp_norm", 0.0))

            for k, arr in zip(keys, arrays):
                weighted_deltas[k] += arr.astype(np.float64) * num_examples

            total_examples += num_examples

        new_state = {}
        any_nonfinite = False
        for k in keys:
            avg_delta = weighted_deltas[k] / max(1, total_examples)
            if not np.all(np.isfinite(avg_delta)):
                any_nonfinite = True
                avg_delta = np.zeros_like(avg_delta)
            new_state[k] = global_state[k] + torch.tensor(avg_delta, dtype=global_state[k].dtype, device=global_state[k].device)

        if any_nonfinite:
            print(f"WARNING: round {server_round} produced non-finite aggregated delta for one or more "
                  f"tensors -- those tensors were left unchanged this round instead of applying garbage.")

        self.global_model.load_state_dict(new_state)
        aggregated_params = ndarrays_to_parameters([v.cpu().numpy() for v in new_state.values()])

        acc = self.evaluate_global(final=False)
        avg_pre = float(np.mean(pre_norms)) if pre_norms else 0.0
        avg_post = float(np.mean(post_norms)) if post_norms else 0.0
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f} | "
              f"Avg pre-clip norm: {avg_pre:.4f} | Avg post-DP norm: {avg_post:.4f}")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), f"best_model_dp_{PRIVACY_LEVEL}.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)

        return aggregated_params, {"accuracy": acc, "avg_pre_clip_norm": avg_pre, "avg_post_dp_norm": avg_post}

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
        plt.title(f"Final Confusion Matrix (DP-FedAvg, {PRIVACY_LEVEL} privacy)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"final_confusion_matrix_dp_{PRIVACY_LEVEL}.png")
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
        use_dp=USE_DP,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=NOISE_MULTIPLIER,
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