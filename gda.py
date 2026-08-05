"""
Gradient-inversion attack harness (v3): empirically measures how hard it
is to reconstruct a client's raw IMU input window from what the server
actually receives, sweeping across:

    1. NO PRIVACY     -- raw single local-training-step gradient
    2. SAPM           -- Fisher-sparsified + quantized + permuted, then
                         unpermuted/dequantized exactly as aggregate_fit
                         does, at EACH keep_ratio in SAPM_KEEP_RATIOS
    3. DP-FEDAVG      -- L2-clipped + Gaussian-noised gradient, at EACH
                         privacy level in DP_PRIVACY_LEVELS (same presets
                         as fl_train_dp_strong.py)
    4. SECAGG + TOP-K -- what secagg_topk.py's server actually ever sees,
                         at EACH keep_fraction in SECAGG_TOPK_FRACTIONS
                         (NEW in this version -- see below)

THREAT MODEL: the attacker is the aggregating server (has the model
weights and, for SAPM/SecAgg, the round-derived seed/support).

WHY SECAGG NEEDS A DIFFERENT KIND OF SCENARIO THAN SAPM/DP:
    SAPM and DP-FedAvg are both SINGLE-CLIENT transforms: the server
    receives one (sparsified/quantized, or clipped/noised) gradient per
    client, so "attack the thing the server received" means attacking
    one transformed victim gradient -- exactly what this harness already
    did. SecAgg is fundamentally different: by design, the server NEVER
    receives any individual client's gradient at all, only the SUM
    across SECAGG_NUM_CLIENTS clients (with the pairwise masks having
    cancelled out exactly). There is no single "the server received this
    for the victim" object to attack -- the closest honest analogue is
    "the server received this for the victim's ROUND", which is an
    average over multiple clients.

    This version adds TWO complementary scenarios per keep_fraction,
    matching secagg_topk.py's two actual layers of defense:

    - "secagg_kr{X}_single_masked": what ONE client's raw wire message
      looks like -- the victim's own K-sparse gradient plus i.i.d.
      Gaussian mask noise (std=SECAGG_MASK_SCALE, matching MASK_SCALE in
      secagg_topk.py). This is what an attacker would see if they could
      intercept a single client's transmission on the wire BEFORE the
      sum cancels the masks (e.g. a passive network eavesdropper, or a
      bug in the mask-cancellation logic like the ones found while
      debugging secagg_topk.py -- see that file's module docstring).
      Tests whether the masking ALONE protects an individual message.

    - "secagg_kr{X}_aggregate": the exact value the server legitimately
      recovers after correct unmasking -- the TRUE AVERAGE of the
      K-sparse gradient across SECAGG_NUM_CLIENTS participants that
      round (no noise at all; the whole point of SecAgg is that the mask
      cancels exactly). The other (SECAGG_NUM_CLIENTS - 1) participants
      are real gradients from OTHER randomly-sampled training windows,
      run through the SAME current model, standing in for the other
      real clients present that round. Tests whether AVERAGING with
      other clients alone protects the victim, with the cryptography set
      aside entirely (i.e. this measures the best case for an attacker
      who has somehow already defeated SecAgg's masking and is now
      looking at the legitimate plaintext aggregate).

    Both use the SAME round-synchronized support S (every scenario's
    participants must use identical K indices for the "aggregate" case
    to even make sense, mirroring the real requirement in secagg_topk.py
    that all clients mask/sparsify to the same support each round).

    Neither scenario is "the" SecAgg threat model on its own -- together
    they bracket it: "single_masked" is the worst case if the crypto
    fails, "aggregate" is the best case if the crypto holds perfectly.
    Comparing both against SAPM/DP (which offer no averaging-across-
    clients defense at all, only per-client transforms) is the fairest
    way to see what SecAgg's aggregation actually buys you beyond
    sparsification alone.

NEW IN THIS VERSION (carried over from v2):
- Sweeps several SAPM/DP/SecAgg settings instead of one point each, so
  you get an actual privacy-vs-reconstruction curve for each mechanism.
- PROTECT_FINAL_LAYER: when True, the final classification layer's
  weight+bias are never transmitted at all (for every scenario,
  including both new SecAgg ones -- their indices are simply excluded
  from the round's support) -- lets you directly test whether this
  closes the label-leak side channel.
- Paired Wilcoxon signed-rank test (scenario vs. no_privacy) on MSE,
  since the same victims are used across all scenarios -- gives you a
  p-value instead of an eyeballed mean difference.
- Saves a raw per-victim, per-scenario CSV (gradient_inversion_results.csv)
  for your own plots later.

RUNTIME WARNING: this sweeps multiple settings, which multiplies cost.
With defaults below (NUM_VICTIMS x (1 no-privacy + 3 SAPM + 4 DP + 3*2
SecAgg) = 14 scenarios x 400 attack iterations), expect this to take
noticeably longer than v1/v2. Lower NUM_VICTIMS or the sweep lists first
if you just want a quick smoke test. The SecAgg scenarios additionally
require computing (SECAGG_NUM_CLIENTS - 1) extra gradients per victim
(the co-participants) -- cheap relative to the 400-iteration attack
itself, but not free.

CAVEATS (same as before -- read before treating results as a formal
privacy proof): this is an empirical demonstration using a single
local-training-step gradient against a strong (server) attacker with
400 optimization iterations; results depend on those choices and should
be reported as such, not as a formal guarantee. For SecAgg specifically:
the "aggregate" scenario assumes the mask cancellation itself is
correct (a real, separately-verified property of secagg_topk.py, not
re-derived here) -- this script does not re-simulate the ECDH/HKDF/mask
machinery, it directly constructs the plaintext value that correct
cancellation is defined to produce, since that machinery's correctness
is a cryptographic property, not a statistical one this kind of
empirical attack could usefully test.
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

# --- SecAgg + Top-K sweep (mirrors secagg_topk.py's constants exactly) ---
SECAGG_TOPK_FRACTIONS = [0.1, 0.3, 0.6]   # same points as SAPM_KEEP_RATIOS,
                                            # for a direct side-by-side.
SECAGG_NUM_CLIENTS = 3                     # NUM_CLIENTS in secagg_topk.py --
                                            # how many clients' K-sparse
                                            # updates get averaged together
                                            # before the server ever sees
                                            # anything (the "aggregation"
                                            # defense SAPM/DP don't have).
SECAGG_MASK_SCALE = 10.0                   # MASK_SCALE in secagg_topk.py --
                                            # std of the pairwise mask noise
                                            # on a single client's WIRE
                                            # message, before cancellation.

# --- Final-layer protection toggle ---
PROTECT_FINAL_LAYER = True   # set True to test whether excluding the final
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


# ====================== SECAGG + TOP-K TRANSFORM ======================
def flatten_grad_dict(grad_dict, param_names):
    """Flatten a {name: tensor} gradient dict into one flat float32 vector,
    in the SAME concatenation order secagg_topk.py uses (iteration order
    of named_parameters()) -- required for the round-synchronized support
    indices to mean the same thing across participants."""
    arrays = [grad_dict[n].detach().cpu().numpy().astype(np.float32) for n in param_names]
    shapes = [a.shape for a in arrays]
    flat = np.concatenate([a.reshape(-1) for a in arrays])
    return flat, shapes

def unflatten_to_dict(flat, param_names, shapes, ref_dict):
    out = {}
    offset = 0
    for n, shape in zip(param_names, shapes):
        size = int(np.prod(shape))
        arr = flat[offset:offset + size].reshape(shape)
        out[n] = torch.tensor(arr, dtype=ref_dict[n].dtype, device=ref_dict[n].device)
        offset += size
    return out

def secagg_support_indices(total_dim, k, seed):
    """Same idea as support_for_round() in secagg_topk.py: a public,
    deterministic pseudo-random K-subset. The seed here stands in for
    that file's round number -- what matters for this harness is just
    that every participant in a simulated "round" uses the SAME support,
    which a shared seed guarantees."""
    rng = np.random.RandomState(seed % (2 ** 31 - 1))
    return np.sort(rng.choice(total_dim, size=k, replace=False))

def _protected_index_mask(param_names, shapes, total_dim):
    offset = 0
    protected = np.zeros(total_dim, dtype=bool)
    for n, shape in zip(param_names, shapes):
        size = int(np.prod(shape))
        if n in FINAL_LAYER_KEYS:
            protected[offset:offset + size] = True
        offset += size
    return protected

def apply_secagg_topk_transform(model, true_grads, other_grad_dicts, keep_fraction,
                                 mask_scale, protect_final_layer, seed):
    """Builds BOTH SecAgg+Top-K scenario variants for one keep_fraction --
    see the module docstring for what each represents and why two are
    needed (SecAgg has no single "the server received this" object the
    way SAPM/DP do). Returns (single_masked_dict, aggregate_dict)."""
    param_names = [n for n, _ in model.named_parameters()]
    victim_flat, shapes = flatten_grad_dict(true_grads, param_names)
    total_dim = victim_flat.size
    k = max(1, int(keep_fraction * total_dim))
    support = secagg_support_indices(total_dim, k, seed)

    if protect_final_layer:
        protected = _protected_index_mask(param_names, shapes, total_dim)
        support = support[~protected[support]]

    # ---- Variant 1: "single_masked" -- one client's raw wire message ----
    # Victim's own K-sparse values, plus i.i.d. Gaussian mask noise at
    # those same K positions (std=mask_scale, matching MASK_SCALE). This
    # is what a network eavesdropper -- or a cancellation-failure bug --
    # would actually see for a single client's transmitted message.
    single_masked_flat = np.zeros(total_dim, dtype=np.float32)
    single_masked_flat[support] = victim_flat[support]
    single_masked_flat[support] += np.random.normal(
        0.0, mask_scale, size=support.size
    ).astype(np.float32)
    single_masked = unflatten_to_dict(single_masked_flat, param_names, shapes, true_grads)

    # ---- Variant 2: "aggregate" -- the true post-cancellation average ----
    # The victim's gradient plus (SECAGG_NUM_CLIENTS - 1) other real
    # participants' gradients (computed on the SAME current model, from
    # OTHER randomly sampled training windows), all restricted to the
    # SAME support, averaged exactly -- no noise, since SecAgg's whole
    # point is that the mask cancels perfectly when done correctly.
    participant_flats = [victim_flat]
    for gd in other_grad_dicts:
        f, _ = flatten_grad_dict(gd, param_names)
        participant_flats.append(f)
    summed = np.zeros(total_dim, dtype=np.float64)
    for f in participant_flats:
        summed[support] += f[support]
    avg_flat = np.zeros(total_dim, dtype=np.float32)
    avg_flat[support] = (summed[support] / len(participant_flats)).astype(np.float32)
    aggregate = unflatten_to_dict(avg_flat, param_names, shapes, true_grads)

    return single_masked, aggregate


def sample_co_participant_grads(model, train_dataset, exclude_idx, num_needed, device):
    """Draws num_needed OTHER random training windows (never the victim's
    own), runs each through the SAME current model for one local-training-
    step gradient, standing in for the other real clients present in a
    SecAgg round. Uses fresh random indices each call -- a fresh set of
    co-participants per victim/keep_fraction, matching how a real round's
    other participants would differ round to round."""
    n = len(train_dataset)
    grads_list = []
    tried = set([exclude_idx])
    while len(grads_list) < num_needed:
        j = np.random.randint(n)
        if j in tried:
            continue
        tried.add(j)
        sample = train_dataset[j]
        imu_raw = sample["imu"]
        imu_t = imu_raw if torch.is_tensor(imu_raw) else torch.tensor(imu_raw)
        imu = imu_t.unsqueeze(0).to(device).float()
        label_val = sample["label"].item() if torch.is_tensor(sample["label"]) else int(sample["label"])
        label = torch.tensor([label_val], device=device, dtype=torch.long)

        model.zero_grad()
        output = model({"imu": imu})
        loss = torch.nn.functional.cross_entropy(output, label)
        grads = torch.autograd.grad(loss, list(model.parameters()))
        grad_dict = {n_: g.detach().clone() for (n_, _), g in zip(model.named_parameters(), grads)}
        grads_list.append(grad_dict)
    return grads_list


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


def build_scenarios(model, true_grads, dp_noise_multipliers, train_dataset, victim_idx):
    scenarios = {"no_privacy": true_grads}
    for kr in SAPM_KEEP_RATIOS:
        scenarios[f"sapm_kr{kr}"] = apply_sapm_transform(true_grads, kr, SAPM_QUANT_BITS, PROTECT_FINAL_LAYER)
    for level in DP_PRIVACY_LEVELS:
        scenarios[f"dp_{level}"] = apply_dp_transform(
            true_grads, DP_MAX_GRAD_NORM, dp_noise_multipliers[level], PROTECT_FINAL_LAYER
        )
    # SecAgg + Top-K: needs the other participants' gradients, computed
    # once per victim (shared across the keep_fraction sweep -- a real
    # round only ever has one fixed set of co-participants; sweeping
    # keep_fraction is a "what if we'd configured K differently that
    # round" comparison, so it's correct to hold participants fixed).
    other_grads = sample_co_participant_grads(
        model, train_dataset, victim_idx, SECAGG_NUM_CLIENTS - 1, DEVICE
    )
    for kr in SECAGG_TOPK_FRACTIONS:
        seed = int(victim_idx * 1_000_003 + round(kr * 1000))  # deterministic per (victim, kr)
        single_masked, aggregate = apply_secagg_topk_transform(
            model, true_grads, other_grads, kr, SECAGG_MASK_SCALE, PROTECT_FINAL_LAYER, seed
        )
        scenarios[f"secagg_kr{kr}_single_masked"] = single_masked
        scenarios[f"secagg_kr{kr}_aggregate"] = aggregate
    return scenarios


def run_victim(model, sample_imu, sample_label, dp_noise_multipliers, train_dataset, victim_idx):
    model.zero_grad()
    output = model({"imu": sample_imu})
    loss = torch.nn.functional.cross_entropy(output, sample_label)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    true_grads = {n: g.detach().clone() for (n, _), g in zip(model.named_parameters(), grads)}

    bias_key = "imu_head.4.bias"
    true_label_val = int(sample_label.item())

    results = {}
    scenarios = build_scenarios(model, true_grads, dp_noise_multipliers, train_dataset, victim_idx)

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
    print(f"SecAgg+Top-K keep_fractions to sweep: {SECAGG_TOPK_FRACTIONS} "
          f"(num_clients={SECAGG_NUM_CLIENTS}, mask_scale={SECAGG_MASK_SCALE})")
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
        results, true_label_val = run_victim(
            model, sample_imu, sample_label, dp_noise_multipliers, train_dataset, int(idx)
        )

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
            print(f"  {scenario:28s}: MSE={r['mse']:.6f}  normMSE={r['normalized_mse']:.4f}  "
                  f"label_leaked={r['label_leaked']}  label_correct={r['label_correct']}")

        if i == 0:
            example_plot_data = (sample_imu.cpu().numpy(), results)

    with open("gradient_inversion_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print("\nSaved raw per-victim results to gradient_inversion_results.csv")

    print("\n========== SUMMARY (averaged over all victims) ==========")
    header = f"{'Scenario':28s} {'Avg MSE':>12s} {'Avg normMSE':>14s} {'Label leak':>12s} {'Label correct':>14s} {'p vs no_priv':>14s}"
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
        print(f"{scenario:28s} {avg_mse:12.6f} {avg_nmse:14.4f} {leak_rate:12.1%} {correct_rate:14.1%} {p_str:>14s}")

    print("Interpretation: higher MSE = harder to reconstruct = stronger empirical privacy.")
    print("p < 0.05 vs no_privacy means that scenario's MSE is statistically distinguishable from doing nothing "
          "(with only NUM_VICTIMS paired samples -- increase NUM_VICTIMS for a more defensible p-value).")
    print("For the two 'secagg_kr{X}_*' rows per keep_fraction: 'single_masked' isolates the masking defense "
          "alone (one client's raw wire message); 'aggregate' isolates the averaging-across-clients defense "
          "alone (the plaintext value correct unmasking legitimately produces). Real secagg_topk.py gets BOTH "
          "at once -- comparing each to SAPM/DP at the same keep_fraction shows what SecAgg's extra averaging "
          "step buys you beyond sparsification alone.")
    if not _SCIPY_AVAILABLE:
        print("(Install scipy for the paired significance test: pip install scipy)")

    if example_plot_data is not None:
        true_imu, results = example_plot_data
        channel = 0
        plt.figure(figsize=(16, 8))
        plt.plot(true_imu[0, :, channel], label="True signal", linewidth=2.5, color="black")
        for scenario, r in results.items():
            plt.plot(r["reconstruction"][0, :, channel], label=f"Reconstructed ({scenario})", alpha=0.6)
        plt.title(f"Gradient-inversion reconstruction, IMU channel {channel} (victim 1)")
        plt.xlabel("Timestep")
        plt.ylabel("Sensor value")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig("gradient_inversion_comparison.png")
        plt.close()
        print("Saved comparison plot to gradient_inversion_comparison.png")


if __name__ == "__main__":
    main("train.csv")