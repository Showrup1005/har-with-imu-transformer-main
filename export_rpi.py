import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---- Load the saved run data ----
data = np.load("fl_run_history.npz")

rounds = data["round"]
num_rounds = int(data["num_rounds"]) if "num_rounds" in data.files else int(rounds.max())

# Early-stopping info (absent in runs saved by the older script)
best_round = int(data["best_round"]) if "best_round" in data.files else -1
stopped_round = int(data["stopped_round"]) if "stopped_round" in data.files else -1

# ================= communication cost vs. baseline =================
plt.figure(figsize=(9, 5))
dense_mb = data["comm_dense_bytes"] / 1e6
baseline_mb = data["comm_no_compression_bytes"] / 1e6
width = 0.4
x = np.arange(len(rounds))
plt.bar(x - width / 2, baseline_mb, width, label="No-compression baseline", color="#898781")
plt.bar(x + width / 2, dense_mb, width, label="Actually sent", color="#2a78d6")
tick_step = max(1, len(rounds) // 10)
plt.xticks(x[::tick_step], [rounds[i] for i in range(0, len(rounds), tick_step)])
plt.xlabel("Round"); plt.ylabel("MB per round")
plt.title("Communication cost vs. no-compression baseline")
plt.legend(); plt.tight_layout()
plt.show()

# ================= accuracy over rounds (val + test) =================
plt.figure(figsize=(9, 4))
if "val_accuracy" in data.files:
    plt.plot(rounds, data["val_accuracy"], color="#1baf7a", linewidth=2, label="Validation (drives stopping)")
plt.plot(rounds, data["accuracy"], color="#2a78d6", linewidth=2, label="Test (logged only)")
if best_round > 0:
    plt.axvline(best_round, color="#898781", linestyle=":", linewidth=1.5, label=f"Best-val round ({best_round})")
if stopped_round > 0:
    plt.axvline(stopped_round, color="#d62a2a", linestyle="--", linewidth=1.5, label=f"Early stop ({stopped_round})")
plt.xlim(1, num_rounds)
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
plt.title("Global model accuracy over training rounds")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ================= bit-width per round =================
plt.figure(figsize=(9, 4))
plt.step(rounds, data["num_bits"], where="post", color="#eb6834", linewidth=2)
if stopped_round > 0:
    plt.axvline(stopped_round, color="#d62a2a", linestyle="--", linewidth=1.5, label=f"Early stop ({stopped_round})")
    plt.legend()
plt.xlabel("Round"); plt.ylabel("Bits per element")
plt.title("Quantization bit-width used per round")
plt.tight_layout()
plt.show()

# ================= compute overhead =================
plt.figure(figsize=(7, 4))
overhead_totals = [float(data["total_transform_time_sec"]), float(data["total_reconstruct_time_sec"])]
labels_ov = ["Client transform\n(encode)", "Server reconstruct\n(decode)"]
plt.barh(labels_ov, overhead_totals, color=["#2a78d6", "#1baf7a"])
plt.xlabel("Seconds, summed over the run")
plt.title("Compute overhead: client vs. server")
plt.tight_layout()
plt.show()

# ================= (bonus): quantization MSE over rounds =================
if "quant_mse" in data.files:
    plt.figure(figsize=(9, 4))
    plt.plot(rounds, data["quant_mse"], color="#8a4fd8", linewidth=2)
    plt.xlabel("Round"); plt.ylabel("Mean squared error")
    plt.title("Quantization error over training rounds")
    plt.tight_layout()
    plt.show()

# ================= Confusion matrix (raw + normalized) =================
# These come from the final test evaluation (best-val model if RESTORE_BEST was on).
if data["final_labels"].size > 0:
    y_true = data["final_labels"]
    y_pred = data["final_preds"]
    y_probs = data["final_probs"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Final Confusion Matrix (raw counts)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()

    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Final Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()

    # ================= ROC curves (one-vs-rest, per class) =================
    classes = np.unique(y_true)
    y_onehot = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(9, 8))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (one-vs-rest, per class)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

# ================= Summary =================
if "best_val_acc" in data.files:
    print(f"\nBest validation accuracy: {float(data['best_val_acc']):.4f} (round {best_round})")
    print(f"Test accuracy at best-val round: {float(data['best_acc']):.4f}")
else:
    print(f"\nBest accuracy achieved: {float(data['best_acc']):.4f}")

if stopped_round > 0:
    print(f"Early stopped at round {stopped_round} of {num_rounds}")
else:
    print(f"Ran all {num_rounds} rounds (no early stop)")

if data["total_comm_no_compression_bytes"] > 0:
    ratio = float(data["total_comm_dense_bytes"]) / float(data["total_comm_no_compression_bytes"]) * 100
    print(f"Real compression achieved: {ratio:.1f}% of no-compression baseline "
          f"(over the {len(rounds)} rounds that ran)")