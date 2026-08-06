"""
dp_calibrate.py

Fast, offline sweep tool to find sane (keep_ratio, clip_norm, epsilon)
combinations WITHOUT re-running the full 40-round FL simulation for
every candidate.

HOW TO GET THE INPUTS:
    Run the instrumented fl_train.py for just 1-3 rounds at your
    CURRENT keep_ratio (e.g. 0.3). Read the printed lines:
        [dp] k_total=... 
        [dp-calib] avg pre-clip L2 norm of selected values: X
    Plug k_total and X into REFERENCE_K and REFERENCE_RAW_NORM below.

WHAT THIS ESTIMATES:
    If you change keep_ratio, dp_k_total changes roughly linearly
    (k ~ keep_ratio * D, since D is fixed by the model size). This
    script projects how the pre-clip L2 norm would change under a
    different keep_ratio using a CONSERVATIVE heuristic:

        raw_norm(k) ~ REFERENCE_RAW_NORM * sqrt(k / REFERENCE_K)

    This assumes coordinate magnitudes stay roughly constant as you
    shrink k (L2 norm ~ sqrt(n) * mean magnitude for iid-ish entries).
    In reality SAPM's top-k selection is a Fisher-sensitivity ranking,
    so as you shrink keep_ratio you keep progressively LARGER-magnitude
    coordinates -- meaning the real raw_norm shrinks SLOWER than this
    sqrt(k) estimate predicts. That makes this projection a lower
    bound / pessimistic estimate of achievable SNR: if the sweep says
    a setting looks good, it will likely look at least as good in
    practice; if it says a setting looks marginal, it's worth trying
    anyway before ruling it out.

    This is a planning aid to prioritize which 2-3 configs are worth
    an actual full training run -- not a substitute for validating the
    winning config with real training.
"""

import numpy as np
import itertools

# ---- Fill these in from a real instrumented run (see docstring) ----
REFERENCE_K = 460401          # dp_k_total observed at REFERENCE_KEEP_RATIO
REFERENCE_KEEP_RATIO = 0.3    # the keep_ratio that produced REFERENCE_K
REFERENCE_RAW_NORM = 1.46     # avg dp_raw_norm observed at that keep_ratio
                               # (placeholder: 1.0 / 0.6856 from your log's
                               # round-1 dp_scale, since raw_norm = clip_norm
                               # / dp_scale when clipping was active -- REPLACE
                               # with the actual printed [dp-calib] value)
DELTA = 1e-5

# ---- What to sweep ----
KEEP_RATIOS = [0.3, 0.1, 0.05, 0.02, 0.01]
EPSILONS = [8.0, 15.0, 25.0, 40.0]


def gaussian_sigma(clip_norm, epsilon, delta):
    return clip_norm * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def project_raw_norm(keep_ratio):
    k = max(1, int(keep_ratio / REFERENCE_KEEP_RATIO * REFERENCE_K))
    # sqrt(k) scaling -- see docstring caveat: this UNDERESTIMATES how much
    # norm survives shrinkage, since top-k keeps the largest-magnitude
    # coordinates first.
    return REFERENCE_RAW_NORM * np.sqrt(k / REFERENCE_K), k


def main():
    print(f"{'keep_ratio':>10} | {'proj_k':>8} | {'proj_raw_norm':>13} | "
          f"{'epsilon':>7} | {'clip_norm*':>10} | {'sigma':>10} | {'proj_snr':>9}")
    print("-" * 90)

    results = []
    for keep_ratio in KEEP_RATIOS:
        proj_norm, k = project_raw_norm(keep_ratio)
        # Use clip_norm == projected raw_norm (no clipping loss, dp_scale~1.0)
        clip_norm = proj_norm
        # Projected per-coordinate signal magnitude ~ raw_norm / sqrt(k)
        signal_mean_abs = proj_norm / np.sqrt(k)

        for epsilon in EPSILONS:
            sigma = gaussian_sigma(clip_norm, epsilon, DELTA)
            snr = signal_mean_abs / sigma if sigma > 0 else float("inf")
            results.append((keep_ratio, k, proj_norm, epsilon, clip_norm, sigma, snr))
            print(f"{keep_ratio:>10.3f} | {k:>8d} | {proj_norm:>13.4f} | "
                  f"{epsilon:>7.1f} | {clip_norm:>10.4f} | {sigma:>10.4f} | {snr:>9.4f}")

    print("\n* clip_norm here is set equal to the projected raw_norm (i.e. the "
          "clip barely engages -- dp_scale ~1.0). If you use a smaller clip_norm "
          "than this, sigma drops but you also start throwing away real signal; "
          "there's a genuine trade-off to explore near this value, not just below it.")

    best = max(results, key=lambda r: r[-1])
    print(f"\nBest projected SNR in this grid: keep_ratio={best[0]}, epsilon={best[3]}, "
          f"clip_norm≈{best[4]:.4f} -> projected snr≈{best[6]:.4f}")
    print("Compare this to your CURRENT run's real snr=0.003 (from the log) -- "
          "anything meaningfully above ~1.0 is where noise stops dominating signal; "
          "aim well above that (e.g. 3-10) for the aggregate to carry useful gradient information.")
    print("\nNext step: pick the top 2-3 rows here, run the REAL instrumented "
          "fl_train.py for each (a handful of rounds is enough to sanity check "
          "the actual dp_snr and accuracy trend before committing to a full 40-round run).")


if __name__ == "__main__":
    main()