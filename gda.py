"""
Gradient-inversion attack harness (v2): empirically measures how hard it
is to reconstruct a client's raw IMU input window from what the server
actually receives, sweeping across:

    1. NO PRIVACY     -- raw single local-training-step gradient
    2. SAPM           -- Fisher-sparsified + quantized + permuted, then
                         unpermuted/dequantized exactly as aggregate_fit
                         does, at EACH keep_ratio in SAPM_KEEP_RATIOS
    3. DP-FEDAVG      -- L2-clipped + Gaussian-noised gradient, at EACH
                         privacy level in DP_PRIVACY_LEVELS (same presets
                         as fl_train_dp_strong.py)

THREAT MODEL: the attacker is the aggregating server (has the model
weights and, for SAPM, the round seed).

NEW IN THIS VERSION:
- Sweeps several SAPM/DP settings instead of one point each, so you get
  an actual privacy-vs-reconstruction curve for both mechanisms.
- PROTECT_FINAL_LAYER: when True, the final classification layer's
  weight+bias are never transmitted at all (for every scenario) -- lets
  you directly test whether this closes the label-leak side channel.
- Paired Wilcoxon signed-rank test (scenario vs. no_privacy) on MSE,
  since the same victims are used across all scenarios -- gives you a
  p-value instead of an eyeballed mean difference.
- Saves a raw per-victim, per-scenario CSV (gradient_inversion_results.csv)
  for your own plots later.

RUNTIME WARNING: this sweeps multiple settings, which multiplies cost.
With defaults below (10 victims x (1 no-privacy + 3 SAPM + 4 DP) = 8
scenarios x 400 attack iterations), expect this to take noticeably
longer than the single-setting v1 script. Lower NUM_VICTIMS or the
sweep lists first if you just want a quick smoke test.

CAVEATS (same as before -- read before treating results as a formal
privacy proof): this is an empirical demonstration using a single
local-training-step gradient against a strong (server) attacker with
400 optimization iterations; results depend on those choices and should
be reported as such, not as a formal guarantee.
"""

import csv
import json
import math
import warnings
import contextlib
import numpy as np
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import scipy.stats  # noqa: F401
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset


@contextlib.contextmanager
def math_attention_only():
    """DLG-style attacks need to backprop through the gradient computation
    itself (a 'double backward'). PyTorch's fused/flash/memory-efficient
    attention kernels don't implement that second-order derivative -- this
    forces the plain 'math' SDPA backend, which does."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except ImportError:
        pass
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield
        return
    except AttributeError:
        pass
    yield

# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

MODEL_CHECKPOINT = "best_model.pth"   # None = random init (worst-case); or path to a trained checkpoint

# --- SAPM sweep ---
SAPM_KEEP_RATIOS = [0.1, 0.3, 0.6]
SAPM_QUANT_BITS = 8

# --- DP sweep (same presets as fl_train_dp_strong.py) ---
DP_PRIVACY_LEVELS = ["minimal", "moderate", "strong", "very_strong"]
DP_MAX_GRAD_NORM = 1.0
_DP_LEVEL_MULTIPLIERS = {"minimal": 1, "moderate": 10, "strong": 30, "very_strong": 100}

# --- Final-layer protection toggle ---
PROTECT_FINAL_LAYER = False   # set True to test whether excluding the final
                              # layer from transmission closes the label leak
FINAL_LAYER_KEYS = ["imu_head.4.weight", "imu_head.4.bias"]

# --- Attack optimization ---
ATTACK_ITERS = 400
ATTACK_LR = 0.05
NUM_VICTIMS = 30   


def gaussian_mechanism_epsilon(noise_multiplier, delta):
    return math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier


# ====================== SAPM TRANSFORM ======================
def compute_topk_mask(fisher_flat, keep_ratio):
    n = fisher_flat.size
    k = max(1, int(np.ceil(keep_ratio * n)))
    if k >= n:
        return np.ones(n, dtype=bool)
    idx = np.argpartition(fisher_flat, -k)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def compute_quant_params(x):
    x_min, x_max = float(x.min()), float(x.max())
    if x_max == x_min:
        return 1.0, x_min
    return x_max - x_min, x_min

def quantize_with_params(x, scale, zmin, num_bits=8):
    if num_bits >= 32:
        return x.astype(np.float32)
    qmax = 2 ** num_bits - 1
    step = scale / qmax if scale != 0 else 1.0
    x_scaled = (x - zmin) / step
    floor = np.floor(x_scaled)
    prob = np.clip(x_scaled - floor, 0.0, 1.0)
    rnd = np.random.rand(*x.shape)
    x_q = floor + (rnd < prob)
    return np.clip(x_q, 0, qmax).astype(np.float32)

def dequantize_with_params(x_q, scale, zmin, num_bits=8):
    if num_bits >= 32:
        return x_q
    qmax = 2 ** num_bits - 1
    step = scale / qmax if scale != 0 else 1.0
    return x_q * step + zmin

def apply_sapm_transform(true_grads, keep_ratio, quant_bits, protect_final_layer):
    out = {}
    for name, g in true_grads.items():
        if protect_final_layer and name in FINAL_LAYER_KEYS:
            out[name] = torch.zeros_like(g)
            continue
        g_np = g.cpu().numpy().astype(np.float32)
        flat = g_np.reshape(-1)
        fisher_flat = flat ** 2
        mask = compute_topk_mask(fisher_flat, keep_ratio)
        sparse = np.where(mask, flat, 0.0).astype(np.float32)
        scale, zmin = compute_quant_params(flat)
        q = quantize_with_params(sparse, scale, zmin, quant_bits)
        recon = dequantize_with_params(q, scale, zmin, quant_bits)
        out[name] = torch.tensor(recon.reshape(g_np.shape), dtype=g.dtype, device=g.device)
    return out


# ====================== DP TRANSFORM ======================
def apply_dp_transform(true_grads, max_norm, noise_multiplier, protect_final_layer):
    names = list(true_grads.keys())
    arrays = [true_grads[n].cpu().numpy().astype(np.float32) for n in names]
    pre_norm = float(np.sqrt(sum(np.sum(a.astype(np.float64) ** 2) for a in arrays)))
    clip_factor = min(1.0, max_norm / (pre_norm + 1e-12))
    arrays = [a * clip_factor for a in arrays]
    noise_std = noise_multiplier * max_norm
    out = {}
    for n, a in zip(names, arrays):
        if protect_final_layer and n in FINAL_LAYER_KEYS:
            out[n] = torch.zeros_like(true_grads[n])
            continue
        noised = a + np.random.normal(0.0, noise_std, size=a.shape).astype(np.float32)
        out[n] = torch.tensor(noised, dtype=true_grads[n].dtype, device=true_grads[n].device)
    return out


# ====================== ATTACK ======================
def infer_label_from_bias(target_grads, bias_key, num_classes):
    if bias_key not in target_grads:
        return None, False
    bias_grad = target_grads[bias_key].detach().cpu().numpy()
    if np.allclose(bias_grad, 0.0):
        return None, False
    return int(np.argmin(bias_grad)), True

def gradient_matching_attack(model, target_grads, true_label, input_shape, iters, lr):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    dummy_input = torch.randn(input_shape, device=DEVICE, requires_grad=True)
    label_t = torch.tensor([true_label], device=DEVICE, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([dummy_input], lr=lr)

    param_names = [n for n, _ in model.named_parameters()]
    target_list = [target_grads[n] for n in param_names]

    for _ in range(iters):
        optimizer.zero_grad()
        with math_attention_only():
            output = model({"imu": dummy_input})
            loss = criterion(output, label_t)
            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            match_loss = sum(((g - t) ** 2).sum() for g, t in zip(grads, target_list))
            match_loss.backward()
        optimizer.step()

    return dummy_input.detach()


def build_scenarios(true_grads, dp_noise_multipliers):
    scenarios = {"no_privacy": true_grads}
    for kr in SAPM_KEEP_RATIOS:
        scenarios[f"sapm_kr{kr}"] = apply_sapm_transform(true_grads, kr, SAPM_QUANT_BITS, PROTECT_FINAL_LAYER)
    for level in DP_PRIVACY_LEVELS:
        scenarios[f"dp_{level}"] = apply_dp_transform(
            true_grads, DP_MAX_GRAD_NORM, dp_noise_multipliers[level], PROTECT_FINAL_LAYER
        )
    return scenarios


def run_victim(model, sample_imu, sample_label, dp_noise_multipliers):
    model.zero_grad()
    output = model({"imu": sample_imu})
    loss = torch.nn.functional.cross_entropy(output, sample_label)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    true_grads = {n: g.detach().clone() for (n, _), g in zip(model.named_parameters(), grads)}

    bias_key = "imu_head.4.bias"
    true_label_val = int(sample_label.item())

    results = {}
    scenarios = build_scenarios(true_grads, dp_noise_multipliers)

    for name, target_grads in scenarios.items():
        pred_label, leaked = infer_label_from_bias(target_grads, bias_key, config["num_classes"])
        label_correct = (pred_label == true_label_val) if leaked else False
        attack_label = pred_label if leaked else int(np.random.randint(config["num_classes"]))

        recon = gradient_matching_attack(
            model, target_grads, attack_label, sample_imu.shape, ATTACK_ITERS, ATTACK_LR
        )
        mse = torch.mean((recon - sample_imu) ** 2).item()
        true_var = torch.var(sample_imu).item()
        normalized_mse = mse / (true_var + 1e-12)

        results[name] = {
            "mse": mse,
            "normalized_mse": normalized_mse,
            "label_leaked": leaked,
            "label_correct": label_correct,
            "reconstruction": recon.cpu().numpy(),
        }

    return results, true_label_val


def wilcoxon_p_value(baseline_mses, scenario_mses):
    """Paired Wilcoxon signed-rank test, scenario vs. no_privacy MSE.
    Returns None if scipy isn't installed or all diffs are zero."""
    if not _SCIPY_AVAILABLE:
        return None
    diffs = np.array(scenario_mses) - np.array(baseline_mses)
    if np.allclose(diffs, 0):
        return None
    _, p = scipy.stats.wilcoxon(scenario_mses, baseline_mses)
    return p


def main(train_csv: str):
    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Loaded {len(train_dataset)} training windows for attack evaluation.")

    model = IMUTransformerEncoder(config).to(DEVICE)
    if MODEL_CHECKPOINT:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
        print(f"Attacking a TRAINED model loaded from {MODEL_CHECKPOINT}")
    else:
        print("Attacking a RANDOMLY INITIALIZED model (worst-case / early-training scenario)")

    num_params = sum(p.numel() for p in model.parameters())
    baseline_sigma = 1.0 / (num_params ** 0.5)
    dp_noise_multipliers = {level: baseline_sigma * mult for level, mult in _DP_LEVEL_MULTIPLIERS.items()}

    print(f"Model has {num_params:,} parameters | baseline (minimal) noise_multiplier = {baseline_sigma:.6f}")
    print(f"SAPM keep_ratios to sweep: {SAPM_KEEP_RATIOS} (quant_bits={SAPM_QUANT_BITS})")
    print(f"DP levels to sweep: {DP_PRIVACY_LEVELS}")
    for level in DP_PRIVACY_LEVELS:
        sigma = dp_noise_multipliers[level]
        eps = gaussian_mechanism_epsilon(sigma, 1e-5)
        print(f"  {level:12s} sigma={sigma:.6f}  per-round epsilon (approx)={eps:.2f}")
    print(f"PROTECT_FINAL_LAYER = {PROTECT_FINAL_LAYER}")
    print(f"Attack: {ATTACK_ITERS} iters, lr={ATTACK_LR}, {NUM_VICTIMS} victims\n")

    indices = np.random.choice(len(train_dataset), size=NUM_VICTIMS, replace=False)

    all_scenario_names = None
    all_results = {}
    example_plot_data = None
    csv_rows = []

    for i, idx in enumerate(indices):
        sample = train_dataset[idx]
        imu_raw = sample["imu"]
        imu_t = imu_raw if torch.is_tensor(imu_raw) else torch.tensor(imu_raw)
        sample_imu = imu_t.unsqueeze(0).to(DEVICE).float()
        label_val = sample["label"].item() if torch.is_tensor(sample["label"]) else int(sample["label"])
        sample_label = torch.tensor([label_val], device=DEVICE, dtype=torch.long)

        print(f"--- Victim {i+1}/{NUM_VICTIMS} (true label {label_val}) ---")
        results, true_label_val = run_victim(model, sample_imu, sample_label, dp_noise_multipliers)

        if all_scenario_names is None:
            all_scenario_names = list(results.keys())
            all_results = {name: [] for name in all_scenario_names}

        for scenario, r in results.items():
            all_results[scenario].append(r)
            csv_rows.append({
                "victim": i, "true_label": true_label_val, "scenario": scenario,
                "mse": r["mse"], "normalized_mse": r["normalized_mse"],
                "label_leaked": r["label_leaked"], "label_correct": r["label_correct"],
            })
            print(f"  {scenario:14s}: MSE={r['mse']:.6f}  normMSE={r['normalized_mse']:.4f}  "
                  f"label_leaked={r['label_leaked']}  label_correct={r['label_correct']}")

        if i == 0:
            example_plot_data = (sample_imu.cpu().numpy(), results)

    with open("gradient_inversion_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print("\nSaved raw per-victim results to gradient_inversion_results.csv")

    print("\n========== SUMMARY (averaged over all victims) ==========")
    header = f"{'Scenario':16s} {'Avg MSE':>12s} {'Avg normMSE':>14s} {'Label leak':>12s} {'Label correct':>14s} {'p vs no_priv':>14s}"
    print(header)
    baseline_mses = [r["mse"] for r in all_results["no_privacy"]]
    for scenario, rs in all_results.items():
        avg_mse = np.mean([r["mse"] for r in rs])
        avg_nmse = np.mean([r["normalized_mse"] for r in rs])
        leak_rate = np.mean([r["label_leaked"] for r in rs])
        correct_rate = np.mean([r["label_correct"] for r in rs])
        scenario_mses = [r["mse"] for r in rs]
        p = wilcoxon_p_value(baseline_mses, scenario_mses) if scenario != "no_privacy" else None
        p_str = f"{p:.4f}" if p is not None else ("--" if scenario == "no_privacy" else "n/a (no scipy)")
        print(f"{scenario:16s} {avg_mse:12.6f} {avg_nmse:14.4f} {leak_rate:12.1%} {correct_rate:14.1%} {p_str:>14s}")

    print("Interpretation: higher MSE = harder to reconstruct = stronger empirical privacy.")
    print("p < 0.05 vs no_privacy means that scenario's MSE is statistically distinguishable from doing nothing "
          "(with only NUM_VICTIMS paired samples -- increase NUM_VICTIMS for a more defensible p-value).")
    if not _SCIPY_AVAILABLE:
        print("(Install scipy for the paired significance test: pip install scipy)")

    if example_plot_data is not None:
        true_imu, results = example_plot_data
        channel = 0
        plt.figure(figsize=(14, 7))
        plt.plot(true_imu[0, :, channel], label="True signal", linewidth=2.5, color="black")
        for scenario, r in results.items():
            plt.plot(r["reconstruction"][0, :, channel], label=f"Reconstructed ({scenario})", alpha=0.6)
        plt.title(f"Gradient-inversion reconstruction, IMU channel {channel} (victim 1)")
        plt.xlabel("Timestep")
        plt.ylabel("Sensor value")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("gradient_inversion_comparison.png")
        plt.close()
        print("Saved comparison plot to gradient_inversion_comparison.png")


if __name__ == "__main__":
    main("train.csv")