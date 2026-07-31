"""
Gradient-inversion attack harness: empirically measures how hard it is to
reconstruct a client's raw IMU input window from what the server actually
receives, under three conditions:

    1. NO PRIVACY   -- raw single local-training-step gradient
    2. SAPM         -- Fisher-sparsified + quantized + permuted, then
                       unpermuted/dequantized exactly as your aggregate_fit
                       does (i.e. what the server actually reconstructs)
    3. DP-FEDAVG    -- L2-clipped + Gaussian-noised gradient

This uses the DLG (Deep Leakage from Gradients) / iDLG attack: the
attacker starts from random noise as a guess for the victim's input,
then repeatedly adjusts that guess so that its own gradient (computed by
forward/backward-passing the guess through the SAME model weights the
victim used) matches the gradient the attacker actually observed. If the
match is close, the guess converges toward the true input.

THREAT MODEL: the attacker is the aggregating server (has the model
weights and, for SAPM, the round seed -- the realistic strong-adversary
case established in our earlier discussion, since the permutation gives
no protection against the server itself).

CAVEATS (read before treating results as a formal privacy proof):
- This is an empirical demonstration, not a formal privacy guarantee.
  A low reconstruction error is strong evidence of weak privacy; a high
  reconstruction error is evidence the mechanism raises the practical
  cost of attack, not proof that reconstruction is impossible under any
  attacker/optimizer/prior.
- Uses a SINGLE local-training-step gradient (one forward/backward pass,
  no optimizer.step() applied yet) rather than a multi-epoch delta --
  this is deliberately the strongest, most literature-standard attack
  scenario (multi-step deltas are already harder to invert, which is
  itself worth a sentence in your thesis).
- Results depend on attack hyperparameters (iterations, learning rate,
  init). Run with a few different random victims/seeds and report the
  average/spread, not a single number, for a defensible thesis claim.
- Uses random model weights (SAPM/DP tend to be *more* vulnerable to
  this kind of attack early in training and *less* vulnerable on a
  well-trained model).
"""

import json
import warnings
import contextlib
import numpy as np
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from models.IMUTransformerEncoder import IMUTransformerEncoder
from util.IMUDataset import IMUDataset


@contextlib.contextmanager
def math_attention_only():
    """DLG-style attacks need to backprop through the gradient computation
    itself (a 'double backward'). PyTorch's fused/flash/memory-efficient
    scaled-dot-product-attention kernels (used internally by
    nn.TransformerEncoderLayer / nn.MultiheadAttention) don't implement
    that second-order derivative and raise a RuntimeError. The plain
    'math' SDPA backend does support it -- this context manager forces
    that backend for the duration of the attack."""
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
    # Fallback: no kernel-selection API available on this torch version --
    # proceed without forcing (may still fail on some torch versions/models).
    yield

# ====================== CONFIG ======================
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

MODEL_CHECKPOINT = None   

# --- SAPM parameters  ---
SAPM_KEEP_RATIO = 0.6
SAPM_QUANT_BITS = 8

# --- DP parameters  ---
DP_MAX_GRAD_NORM = 1.0
DP_NOISE_MULTIPLIER = None  # None = auto-calibrate from param count, as in fl_train_dp.py

# --- Attack optimization ---
ATTACK_ITERS = 400
ATTACK_LR = 0.05
NUM_VICTIMS = 5   


# ====================== SAPM TRANSFORM (same math as fl_train_sapm.py) ======================
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

def apply_sapm_transform(true_grads, keep_ratio, quant_bits):
    """Simulates the FULL round trip: client-side mask+quantize+permute,
    then server-side unpermute+dequantize -- returns what the server
    actually reconstructs and uses (a client-supplied seed isn't even
    needed here since permutation is a no-op against the server: shuffle
    then un-shuffle with the correct seed always returns the original
    order, so we skip simulating it explicitly)."""
    out = {}
    for name, g in true_grads.items():
        g_np = g.cpu().numpy().astype(np.float32)
        flat = g_np.reshape(-1)
        fisher_flat = flat ** 2  # single-step Fisher proxy = grad^2
        mask = compute_topk_mask(fisher_flat, keep_ratio)
        sparse = np.where(mask, flat, 0.0).astype(np.float32)
        scale, zmin = compute_quant_params(flat)
        q = quantize_with_params(sparse, scale, zmin, quant_bits)
        recon = dequantize_with_params(q, scale, zmin, quant_bits)
        out[name] = torch.tensor(recon.reshape(g_np.shape), dtype=g.dtype, device=g.device)
    return out


# ====================== DP TRANSFORM (same math as fl_train_dp.py) ======================
def apply_dp_transform(true_grads, max_norm, noise_multiplier):
    names = list(true_grads.keys())
    arrays = [true_grads[n].cpu().numpy().astype(np.float32) for n in names]
    pre_norm = float(np.sqrt(sum(np.sum(a.astype(np.float64) ** 2) for a in arrays)))
    clip_factor = min(1.0, max_norm / (pre_norm + 1e-12))
    arrays = [a * clip_factor for a in arrays]
    noise_std = noise_multiplier * max_norm
    arrays = [a + np.random.normal(0.0, noise_std, size=a.shape).astype(np.float32) for a in arrays]
    return {n: torch.tensor(a, dtype=true_grads[n].dtype, device=true_grads[n].device) for n, a in zip(names, arrays)}


# ====================== ATTACK ======================
def infer_label_from_bias(target_grads, bias_key, num_classes):
    """iDLG trick: for cross-entropy loss on a single sample, the true
    class's logit-bias gradient is negative and all others are positive
    (since dL/dz_y = softmax(z)_y - 1 < 0 for the true class, softmax(z)_c
    > 0 otherwise). Returns (predicted_label, leaked) where leaked=False
    means the bias gradient was entirely zeroed out (e.g. by SAPM's
    sparsification) and no label could be inferred from it."""
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


def run_victim(model, sample_imu, sample_label):
    model.zero_grad()
    output = model({"imu": sample_imu})
    loss = torch.nn.functional.cross_entropy(output, sample_label)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    true_grads = {n: g.detach().clone() for (n, _), g in zip(model.named_parameters(), grads)}

    bias_key = "imu_head.4.bias"
    num_classes = config["num_classes"]
    true_label_val = int(sample_label.item())

    results = {}
    scenarios = {
        "no_privacy": true_grads,
        "sapm": apply_sapm_transform(true_grads, SAPM_KEEP_RATIO, SAPM_QUANT_BITS),
        "dp": apply_dp_transform(true_grads, DP_MAX_GRAD_NORM, DP_NOISE_MULTIPLIER),
    }

    for name, target_grads in scenarios.items():
        pred_label, leaked = infer_label_from_bias(target_grads, bias_key, num_classes)
        label_correct = (pred_label == true_label_val) if leaked else False
        attack_label = pred_label if leaked else int(np.random.randint(num_classes))

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


def main(train_csv: str):
    global DP_NOISE_MULTIPLIER

    train_dataset = IMUDataset(train_csv, config["window_size"], config["input_dim"], config["window_shift"])
    print(f"Loaded {len(train_dataset)} training windows for attack evaluation.")

    model = IMUTransformerEncoder(config).to(DEVICE)
    if MODEL_CHECKPOINT:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
        print(f"Attacking a TRAINED model loaded from {MODEL_CHECKPOINT}")
    else:
        print("Attacking a RANDOMLY INITIALIZED model (worst-case / early-training scenario)")

    if DP_NOISE_MULTIPLIER is None:
        num_params = sum(p.numel() for p in model.parameters())
        DP_NOISE_MULTIPLIER = 1.0 / (num_params ** 0.5)
        print(f"Auto-calibrated DP_NOISE_MULTIPLIER = {DP_NOISE_MULTIPLIER:.6f} ({num_params:,} params)")

    print(f"SAPM: keep_ratio={SAPM_KEEP_RATIO}, quant_bits={SAPM_QUANT_BITS}")
    print(f"DP: max_grad_norm={DP_MAX_GRAD_NORM}, noise_multiplier={DP_NOISE_MULTIPLIER}")
    print(f"Attack: {ATTACK_ITERS} iters, lr={ATTACK_LR}, {NUM_VICTIMS} victims\n")

    indices = np.random.choice(len(train_dataset), size=NUM_VICTIMS, replace=False)

    all_results = {"no_privacy": [], "sapm": [], "dp": []}
    example_plot_data = None

    for i, idx in enumerate(indices):
        sample = train_dataset[idx]
        imu_raw = sample["imu"]
        imu_t = imu_raw if torch.is_tensor(imu_raw) else torch.tensor(imu_raw)
        sample_imu = imu_t.unsqueeze(0).to(DEVICE).float()
        label_val = sample["label"].item() if torch.is_tensor(sample["label"]) else int(sample["label"])
        sample_label = torch.tensor([label_val], device=DEVICE, dtype=torch.long)

        print(f"--- Victim {i+1}/{NUM_VICTIMS} (true label {label_val}) ---")
        results, true_label_val = run_victim(model, sample_imu, sample_label)

        for scenario, r in results.items():
            all_results[scenario].append(r)
            print(f"  {scenario:10s}: MSE={r['mse']:.6f}  normMSE={r['normalized_mse']:.4f}  "
                  f"label_leaked={r['label_leaked']}  label_correct={r['label_correct']}")

        if i == 0:
            example_plot_data = (sample_imu.cpu().numpy(), results)

    print("\n========== SUMMARY (averaged over all victims) ==========")
    print(f"{'Scenario':12s} {'Avg MSE':>12s} {'Avg normMSE':>14s} {'Label leak rate':>18s} {'Label correct rate':>20s}")
    for scenario, rs in all_results.items():
        avg_mse = np.mean([r["mse"] for r in rs])
        avg_nmse = np.mean([r["normalized_mse"] for r in rs])
        leak_rate = np.mean([r["label_leaked"] for r in rs])
        correct_rate = np.mean([r["label_correct"] for r in rs])
        print(f"{scenario:12s} {avg_mse:12.6f} {avg_nmse:14.4f} {leak_rate:18.2%} {correct_rate:20.2%}")

    print("\nInterpretation: higher MSE / normalized MSE = harder to reconstruct = stronger empirical "
          "privacy. Lower label-leak/correct rate = stronger empirical privacy against the label side-channel.")

    if example_plot_data is not None:
        true_imu, results = example_plot_data
        channel = 0
        plt.figure(figsize=(12, 6))
        plt.plot(true_imu[0, :, channel], label="True signal", linewidth=2, color="black")
        for scenario, r in results.items():
            plt.plot(r["reconstruction"][0, :, channel], label=f"Reconstructed ({scenario})", alpha=0.7)
        plt.title(f"Gradient-inversion reconstruction, IMU channel {channel} (victim 1)")
        plt.xlabel("Timestep")
        plt.ylabel("Sensor value")
        plt.legend()
        plt.savefig("gradient_inversion_comparison.png")
        plt.close()
        print("\nSaved comparison plot to gradient_inversion_comparison.png")


if __name__ == "__main__":
    main("train.csv")