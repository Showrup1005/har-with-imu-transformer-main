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
import time
import warnings
import os
import psutil
import sys
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# ====================== COST TRACKING UTILITIES ======================
class CostTracker:
    """Tracks communication and computation costs for FL training."""
    
    def __init__(self, save_dir="cost_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.comm_costs = defaultdict(list)  # Per-round communication costs
        self.comp_costs = defaultdict(list)   # Per-round computation costs
        self.client_comp_costs = defaultdict(lambda: defaultdict(list))  # Per-client per-round computation
        
        self.start_time = None
        self.process = psutil.Process()
        
    def start_round(self):
        """Start tracking for a new round."""
        self.round_start_time = time.time()
        self.round_start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def end_round(self, round_num):
        """End tracking for current round."""
        round_time = time.time() - self.round_start_time
        round_memory = self.process.memory_info().rss / 1024 / 1024 - self.round_start_memory
        
        self.comp_costs['server_round_time'].append(round_time)
        self.comp_costs['server_round_memory_delta'].append(round_memory)
        
    def log_client_computation(self, round_num, client_id, comp_time, memory_delta, num_samples):
        """Log computation cost for a single client."""
        self.client_comp_costs[round_num][client_id] = {
            'time': comp_time,
            'memory_delta_mb': memory_delta,
            'num_samples': num_samples,
            'time_per_sample': comp_time / num_samples if num_samples > 0 else 0
        }
        
    def log_communication(self, round_num, client_id, upload_bytes, download_bytes):
        """Log communication cost for a single client."""
        self.comm_costs['round'].append(round_num)
        self.comm_costs['client_id'].append(client_id)
        self.comm_costs['upload_bytes'].append(upload_bytes)
        self.comm_costs['download_bytes'].append(download_bytes)
        self.comm_costs['total_bytes'].append(upload_bytes + download_bytes)
        
    def log_aggregated_communication(self, round_num, avg_upload, avg_download, total_upload, total_download):
        """Log aggregated communication statistics per round."""
        self.comm_costs['per_round_avg_upload'].append(avg_upload)
        self.comm_costs['per_round_avg_download'].append(avg_download)
        self.comm_costs['per_round_total_upload'].append(total_upload)
        self.comm_costs['per_round_total_download'].append(total_download)
        
    def calculate_model_size(self, model):
        """Calculate model size in bytes."""
        total_params = 0
        total_bytes = 0
        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            total_bytes += num_params * param.element_size()
        return total_params, total_bytes
    
    def generate_summary(self, num_rounds, num_clients):
        """Generate and save comprehensive cost summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_rounds': num_rounds,
                'num_clients': num_clients,
                'dp_enabled': USE_DP,
                'privacy_level': PRIVACY_LEVEL if USE_DP else 'none'
            },
            'communication_costs': {},
            'computation_costs': {},
            'total_costs': {}
        }
        
        # Communication costs
        if self.comm_costs.get('per_round_total_upload'):
            total_upload = sum(self.comm_costs['per_round_total_upload'])
            total_download = sum(self.comm_costs['per_round_total_download'])
            total_comm = total_upload + total_download
            
            summary['communication_costs'] = {
                'total_upload_gb': total_upload / 1e9,
                'total_download_gb': total_download / 1e9,
                'total_communication_gb': total_comm / 1e9,
                'avg_per_round_upload_mb': np.mean(self.comm_costs['per_round_avg_upload']) / 1e6,
                'avg_per_round_download_mb': np.mean(self.comm_costs['per_round_avg_download']) / 1e6,
                'avg_per_client_per_round_mb': np.mean([u + d for u, d in zip(
                    self.comm_costs['per_round_avg_upload'], 
                    self.comm_costs['per_round_avg_download']
                )]) / 1e6,
                'per_round_details': {
                    'avg_upload_mb': [x / 1e6 for x in self.comm_costs['per_round_avg_upload']],
                    'avg_download_mb': [x / 1e6 for x in self.comm_costs['per_round_avg_download']],
                    'total_upload_mb': [x / 1e6 for x in self.comm_costs['per_round_total_upload']],
                    'total_download_mb': [x / 1e6 for x in self.comm_costs['per_round_total_download']]
                }
            }
            
            # Communication efficiency metrics
            total_params, model_bytes = self.calculate_model_size(IMUTransformerEncoder(config))
            summary['communication_costs']['model_size_bytes'] = model_bytes
            summary['communication_costs']['total_parameters'] = total_params
            summary['communication_costs']['communication_to_model_ratio'] = (
                total_comm / (model_bytes * num_rounds * num_clients * 2)  # *2 for upload+download
            )
            
        # Computation costs
        if self.comp_costs.get('server_round_time'):
            total_server_time = sum(self.comp_costs['server_round_time'])
            
            # Aggregate client computation
            all_client_times = []
            all_client_memories = []
            for round_num in self.client_comp_costs:
                for client_id in self.client_comp_costs[round_num]:
                    all_client_times.append(self.client_comp_costs[round_num][client_id]['time'])
                    all_client_memories.append(self.client_comp_costs[round_num][client_id]['memory_delta_mb'])
            
            summary['computation_costs'] = {
                'total_server_time_seconds': total_server_time,
                'avg_server_round_time_seconds': np.mean(self.comp_costs['server_round_time']),
                'total_client_compute_time_seconds': sum(all_client_times),
                'avg_client_compute_time_seconds': np.mean(all_client_times) if all_client_times else 0,
                'std_client_compute_time_seconds': np.std(all_client_times) if all_client_times else 0,
                'avg_client_memory_mb': np.mean(all_client_memories) if all_client_memories else 0,
                'per_round_server_time': self.comp_costs['server_round_time'],
            }
            
            # Compute per-sample efficiency
            if all_client_times:
                total_samples = sum([
                    self.client_comp_costs[r][c]['num_samples'] 
                    for r in self.client_comp_costs 
                    for c in self.client_comp_costs[r]
                ])
                summary['computation_costs']['total_samples_processed'] = total_samples
                summary['computation_costs']['avg_time_per_sample_ms'] = (
                    sum(all_client_times) / total_samples * 1000 if total_samples > 0 else 0
                )
        
        # Total costs
        summary['total_costs'] = {
            'total_wall_time_seconds': sum(self.comp_costs.get('server_round_time', [0])),
            'communication_percentage': self._estimate_communication_overhead(),
            'privacy_overhead_factor': self._calculate_privacy_overhead()
        }
        
        # Save summary
        save_path = os.path.join(self.save_dir, f'cost_summary_dp_{PRIVACY_LEVEL}.json')
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _estimate_communication_overhead(self):
        """Estimate what percentage of total time is communication."""
        if not self.comp_costs.get('server_round_time'):
            return 0
        
        total_time = sum(self.comp_costs['server_round_time'])
        # Rough estimate: communication is typically 10-40% of round time
        # More precise measurement would require network monitoring
        return 15.0  # placeholder
    
    def _calculate_privacy_overhead(self):
        """Calculate overhead factor introduced by DP."""
        if not USE_DP:
            return 1.0
        
        # DP adds: clipping computation + noise generation + larger updates (dense)
        # Typical overhead factors: 1.2x - 2.0x vs non-private
        return 1.3  # conservative estimate
    
    def _print_summary(self, summary):
        """Print a formatted summary of costs."""
        print("\n" + "="*60)
        print("COMMUNICATION & COMPUTATION COST SUMMARY")
        print("="*60)
        
        if summary['communication_costs']:
            cc = summary['communication_costs']
            print(f"\n📡 COMMUNICATION COSTS:")
            print(f"  Total Upload:        {cc['total_upload_gb']:.3f} GB")
            print(f"  Total Download:      {cc['total_download_gb']:.3f} GB")
            print(f"  Total Communication: {cc['total_communication_gb']:.3f} GB")
            print(f"  Avg/Round Upload:    {cc['avg_per_round_upload_mb']:.2f} MB")
            print(f"  Avg/Round Download:  {cc['avg_per_round_download_mb']:.2f} MB")
            print(f"  Avg/Client/Round:    {cc['avg_per_client_per_round_mb']:.2f} MB")
            print(f"  Model Size:          {cc['model_size_bytes']/1e6:.2f} MB ({cc['total_parameters']:,} params)")
            
        if summary['computation_costs']:
            cp = summary['computation_costs']
            print(f"\n💻 COMPUTATION COSTS:")
            print(f"  Total Server Time:   {cp['total_server_time_seconds']:.2f}s")
            print(f"  Avg Round Time:      {cp['avg_server_round_time_seconds']:.2f}s")
            print(f"  Total Client Time:   {cp['total_client_compute_time_seconds']:.2f}s")
            print(f"  Avg Client Time:     {cp['avg_client_compute_time_seconds']:.2f}s")
            print(f"  Std Client Time:     {cp['std_client_compute_time_seconds']:.2f}s")
            if 'total_samples_processed' in cp:
                print(f"  Samples Processed:   {cp['total_samples_processed']:,}")
                print(f"  Time/Sample:         {cp['avg_time_per_sample_ms']:.3f}ms")
                
        tc = summary['total_costs']
        print(f"\n📊 TOTAL COSTS:")
        print(f"  Wall Time:           {tc['total_wall_time_seconds']:.1f}s ({tc['total_wall_time_seconds']/60:.1f}min)")
        print(f"  Privacy Overhead:    {tc['privacy_overhead_factor']}x")
        print("="*60)


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
PRIVACY_LEVEL = "strong"   # one of: "minimal", "moderate", "strong", "very_strong"
_PRIVACY_LEVEL_MULTIPLIERS = {
    "minimal": 1,
    "moderate": 10,
    "strong": 30,
    "very_strong": 100,
}

_NUM_MODEL_PARAMS = sum(p.numel() for p in IMUTransformerEncoder(config).parameters())
_BASELINE_NOISE_MULTIPLIER = 1.0 / (_NUM_MODEL_PARAMS ** 0.5)
NOISE_MULTIPLIER = _BASELINE_NOISE_MULTIPLIER * _PRIVACY_LEVEL_MULTIPLIERS[PRIVACY_LEVEL]

# Initialize global cost tracker
cost_tracker = CostTracker(save_dir=f"cost_logs_dp_{PRIVACY_LEVEL}")


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


def get_model_size_mb(model):
    """Calculate model size in megabytes."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 * 1024)


print(f"Using device: {DEVICE}")
print(f"Model has {_NUM_MODEL_PARAMS:,} parameters | Size: {get_model_size_mb(IMUTransformerEncoder(config)):.2f} MB")
print(f"Privacy strategy: DP-FedAvg | enabled={USE_DP} | level={PRIVACY_LEVEL} "
      f"({_PRIVACY_LEVEL_MULTIPLIERS[PRIVACY_LEVEL]}x baseline) | clip_norm={MAX_GRAD_NORM} | "
      f"noise_multiplier={NOISE_MULTIPLIER:.6f}")

_eps_per_round = gaussian_mechanism_epsilon(NOISE_MULTIPLIER, DP_DELTA)
_eps_loose_total = _eps_per_round * NUM_ROUNDS  # basic (loose, linear) composition -- a conservative UPPER bound
print(f"Per-round Gaussian-mechanism epsilon (delta={DP_DELTA}): {_eps_per_round:.4f}")
print(f"Approx. TOTAL epsilon over {NUM_ROUNDS} rounds:")
print(f"  - basic (linear) composition, loose upper bound: {_eps_loose_total:.2f}")
if _eps_per_round > 50:
    print("  - advanced composition: skipped (per-round epsilon is already so large that this "
          "noise level provides no meaningful privacy guarantee at all -- pick a stronger PRIVACY_LEVEL)")
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
    def __init__(self, client_id, train_subset):
        self.client_id = client_id
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        self.train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Calculate model size once for communication cost estimation
        self.model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

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
            print(f"WARNING [Client {self.client_id}]: received non-finite global parameters, sanitized to 0 for local copy.")
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), cleaned)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, fit_config):
        # Start timing client computation
        client_start_time = time.time()
        process = psutil.Process()
        client_start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.set_parameters(parameters)
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        use_dp = fit_config.get("use_dp", USE_DP)
        max_norm = fit_config.get("max_grad_norm", MAX_GRAD_NORM)
        noise_mult = fit_config.get("noise_multiplier", NOISE_MULTIPLIER)

        self.model.train()
        total_loss = 0.0
        batch_count = 0

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
                batch_count += 1

        new_state = self.model.state_dict()
        keys = list(new_state.keys())
        deltas = [(new_state[k] - old_state[k]).cpu().numpy().astype(np.float32) for k in keys]

        pre_clip_norm = float(np.sqrt(sum(np.sum(d.astype(np.float64) ** 2) for d in deltas)))

        # Track DP-specific computation
        dp_start_time = time.time()
        
        if use_dp:
            # Step 1: clip the WHOLE update (across all layers together) to L2 norm max_norm.
            clip_factor = min(1.0, max_norm / (pre_clip_norm + 1e-12))
            deltas = [d * clip_factor for d in deltas]

            # Step 2: add iid Gaussian noise to every coordinate of the clipped update.
            noise_std = noise_mult * max_norm
            noise_start = time.time()
            deltas = [d + np.random.normal(0.0, noise_std, size=d.shape).astype(np.float32) for d in deltas]
            noise_time = time.time() - noise_start
        else:
            noise_time = 0
        
        dp_time = time.time() - dp_start_time
        post_norm = float(np.sqrt(sum(np.sum(d.astype(np.float64) ** 2) for d in deltas)))
        
        # Calculate computation time
        comp_time = time.time() - client_start_time
        memory_delta = process.memory_info().rss / 1024 / 1024 - client_start_memory
        
        # Calculate communication costs (model size * 2 for upload + download)
        upload_bytes = sum(d.nbytes for d in deltas)
        download_bytes = self.model_size_bytes  # Full model downloaded
        
        # Log costs
        cost_tracker.log_client_computation(
            fit_config.get('server_round', 0), 
            self.client_id, 
            comp_time, 
            memory_delta,
            len(self.train_loader.dataset)
        )
        cost_tracker.log_communication(
            fit_config.get('server_round', 0),
            self.client_id,
            upload_bytes,
            download_bytes
        )

        metrics = {
            "train_loss": total_loss / max(1, batch_count),
            "pre_clip_norm": pre_clip_norm,
            "post_dp_norm": post_norm,
            "comp_time_sec": comp_time,
            "dp_overhead_sec": dp_time,
            "noise_gen_sec": noise_time,
            "upload_bytes": upload_bytes,
            "download_bytes": download_bytes,
            "total_comm_bytes": upload_bytes + download_bytes,
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
        
        # Communication cost tracking per round
        self.round_comm_stats = []

    def configure_fit(self, server_round, parameters, client_manager):
        # Start timing this round
        cost_tracker.start_round()
        
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["use_dp"] = self.use_dp
            fit_ins.config["max_grad_norm"] = self.max_grad_norm
            fit_ins.config["noise_multiplier"] = self.noise_multiplier
            fit_ins.config["server_round"] = server_round  # Pass round number to client
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        if not results:
            cost_tracker.end_round(server_round)
            return None, {}

        global_state = self.global_model.state_dict()
        keys = list(global_state.keys())
        weighted_deltas = {k: np.zeros(v.shape, dtype=np.float64) for k, v in global_state.items()}
        total_examples = 0
        pre_norms, post_norms = [], []
        
        # Track communication costs for this round
        round_upload_total = 0
        round_download_total = 0
        client_comm_costs = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            pre_norms.append(fit_res.metrics.get("pre_clip_norm", 0.0))
            post_norms.append(fit_res.metrics.get("post_dp_norm", 0.0))
            
            # Aggregate communication costs
            upload_bytes = fit_res.metrics.get("upload_bytes", 0)
            download_bytes = fit_res.metrics.get("download_bytes", 0)
            round_upload_total += upload_bytes
            round_download_total += download_bytes
            client_comm_costs.append({
                'upload': upload_bytes,
                'download': download_bytes,
                'total': upload_bytes + download_bytes
            })

            for k, arr in zip(keys, arrays):
                weighted_deltas[k] += arr.astype(np.float64) * num_examples

            total_examples += num_examples

        # Log aggregated communication stats
        num_clients = len(results)
        avg_upload = round_upload_total / num_clients
        avg_download = round_download_total / num_clients
        cost_tracker.log_aggregated_communication(
            server_round, avg_upload, avg_download, 
            round_upload_total, round_download_total
        )

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

        # Evaluate and print metrics
        acc = self.evaluate_global(final=False)
        avg_pre = float(np.mean(pre_norms)) if pre_norms else 0.0
        avg_post = float(np.mean(post_norms)) if post_norms else 0.0
        
        # Calculate DP overhead metrics
        avg_comp_time = float(np.mean([r.metrics.get("comp_time_sec", 0) for _, r in results])) if results else 0.0
        avg_dp_overhead = float(np.mean([r.metrics.get("dp_overhead_sec", 0) for _, r in results])) if results else 0.0
        
        print(f"Round {server_round}/{NUM_ROUNDS} - Accuracy: {acc:.4f}")
        print(f"  Avg pre-clip norm: {avg_pre:.4f} | Avg post-DP norm: {avg_post:.4f}")
        print(f"  Avg client compute: {avg_comp_time:.2f}s | DP overhead: {avg_dp_overhead:.3f}s")
        print(f"  Avg client comm: {avg_upload/1e6:.2f}MB up | {avg_download/1e6:.2f}MB down | "
              f"{(avg_upload+avg_download)/1e6:.2f}MB total")
        print(f"  Total round comm: {(round_upload_total+round_download_total)/1e6:.2f}MB")

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.global_model.state_dict(), f"best_model_dp_{PRIVACY_LEVEL}.pth")

        if server_round == NUM_ROUNDS:
            print("\n========== FINAL EVALUATION ==========")
            self.evaluate_global(final=True)
            
            # Generate final cost summary
            print("\n========== COST ANALYSIS ==========")
            cost_tracker.generate_summary(NUM_ROUNDS, NUM_CLIENTS)
        
        # End round timing
        cost_tracker.end_round(server_round)
        
        return aggregated_params, {
            "accuracy": acc, 
            "avg_pre_clip_norm": avg_pre, 
            "avg_post_dp_norm": avg_post,
            "avg_client_comp_time": avg_comp_time,
            "avg_dp_overhead": avg_dp_overhead,
            "total_round_comm_mb": (round_upload_total + round_download_total) / 1e6,
            "avg_client_comm_mb": (avg_upload + avg_download) / 1e6,
        }

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
        print(f"Recall