"""
Reconstruct all standalone evaluation charts and reports from
frontend/standalone_history.json — no retraining required.

Outputs saved to: results/standalone/
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

HISTORY_PATH = "frontend/standalone_history.json"
OUT_DIR      = "results/standalone"


def load():
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def style():
    s = 'seaborn-v0_8-whitegrid'
    plt.style.use(s if s in plt.style.available else 'default')


def main():
    print("=" * 60)
    print("   STANDALONE RESULTS — RECONSTRUCTING FROM JSON")
    print("=" * 60)

    if not os.path.exists(HISTORY_PATH):
        print(f"[ERROR] {HISTORY_PATH} not found.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    d = load()

    epochs          = d["rounds"]                          # 1-50
    accuracy        = d["accuracy"]
    loss            = d["loss"]
    macro_f1        = d["macro_f1"]
    emotion_names   = d["emotion_names"]
    cm              = np.array(d["confusion_matrix_final"])
    per_f1          = d["per_class_f1_final"]
    per_prec        = d["per_class_precision_final"]
    per_rec         = d["per_class_recall_final"]
    clf_report      = d["classification_report"]

    best_acc        = d["best_accuracy"]
    best_mf1        = d["best_macro_f1"]
    best_wf1        = d["best_weighted_f1"]
    best_prec       = d["best_macro_precision"]
    best_rec        = d["best_macro_recall"]
    best_epoch      = d["best_epoch"]
    n_train         = d.get("training_samples", 28709)
    n_test          = d.get("test_samples", 7178)
    n_params        = d.get("model_parameters", 56951)
    lr              = d.get("learning_rate", 0.001)
    bs              = d.get("batch_size", 64)
    n_epochs        = d.get("epochs_total", 50)

    COLORS = ['#ef4444','#22c55e','#3b82f6','#eab308','#6366f1','#f97316','#9ca3af']

    # ── 1. CSV export ─────────────────────────────────────────────────────────
    csv_df = pd.DataFrame({
        "Epoch":           epochs,
        "Val_Accuracy":    accuracy,
        "Val_Loss":        loss,
        "Macro_F1":        macro_f1,
    })
    csv_path = os.path.join(OUT_DIR, "standalone_metrics.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"[1/7] CSV exported            -> {csv_path}")

    # ── 2. Convergence: accuracy + loss ──────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.set_xlabel("Training Epoch", fontweight='bold', labelpad=10)
    ax1.set_ylabel("Validation Accuracy", color='#0084ff', fontweight='bold')
    l1, = ax1.plot(epochs, accuracy, color='#0084ff', marker='o', markersize=3,
                   linewidth=2.5, label="Val Accuracy")
    ax1.axvline(best_epoch, color='#0084ff', linestyle=':', alpha=0.6, linewidth=1.2)
    ax1.annotate(f"Best: {best_acc:.2%}\n(Epoch {best_epoch})",
                 xy=(best_epoch, best_acc), xytext=(best_epoch + 1.5, best_acc - 0.07),
                 fontsize=8.5, color='#0084ff',
                 arrowprops=dict(arrowstyle='->', color='#0084ff', lw=1.2))
    ax1.tick_params(axis='y', labelcolor='#0084ff')
    ax1.set_ylim(0.3, 0.75)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Loss", color='#8a2be2', fontweight='bold')
    l2, = ax2.plot(epochs, loss, color='#8a2be2', marker='x', markersize=3,
                   linestyle='--', linewidth=2.0, label="Val Loss")
    ax2.tick_params(axis='y', labelcolor='#8a2be2')
    ax2.set_ylim(1.0, max(loss) * 1.1)

    plt.title("Standalone mini_XCEPTION — Training Convergence on FER-2013",
              fontsize=12, fontweight='bold', pad=15)
    ax1.legend([l1, l2], ["Val Accuracy", "Val Loss"], loc='lower right')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "standalone_convergence.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[2/7] Convergence plot        -> {out}")

    # ── 3. F1 / Precision / Recall convergence ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(epochs, macro_f1, color='#0084ff', marker='o', markersize=3,
            linewidth=2.5, label="Macro F1")
    # best_prec / best_rec are scalars — draw as horizontal reference lines
    ax.axhline(best_prec, color='#10b981', linestyle=':', linewidth=1.8,
               label=f"Best Macro Precision ({best_prec:.2%})")
    ax.axhline(best_rec,  color='#8a2be2', linestyle=':', linewidth=1.8,
               label=f"Best Macro Recall ({best_rec:.2%})")
    ax.axhline(best_wf1,  color='#f59e0b', linestyle='--', linewidth=1.6,
               label=f"Best Weighted F1 ({best_wf1:.2%})")
    ax.axvline(best_epoch, color='gray', linestyle=':', alpha=0.5, linewidth=1.2)
    ax.set_xlabel("Training Epoch", fontweight='bold', labelpad=10)
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_ylim(0.2, 0.75)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Standalone — F1 / Precision / Recall over Training",
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "standalone_f1_convergence.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[3/7] F1 convergence plot     -> {out}")

    # ── 4. Per-class F1 bar chart (best model) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(emotion_names, per_f1, color=COLORS, edgecolor='white', linewidth=0.8,
                  zorder=3)
    ax.axhline(best_mf1, color='black', linestyle='--', linewidth=1.4,
               label=f"Macro Avg F1 = {best_mf1:.2%}")
    for bar, val in zip(bars, per_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.2%}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("F1 Score", fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(f"Standalone — Per-Class F1 Score (Best Checkpoint — Epoch {best_epoch})",
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "standalone_per_class_f1.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[4/7] Per-class F1 bar        -> {out}")

    # ── 5. Per-class Precision vs Recall grouped bar ──────────────────────────
    x    = np.arange(len(emotion_names))
    w    = 0.28
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w, per_prec, w, label="Precision", color='#3b82f6',
                edgecolor='white', linewidth=0.6, zorder=3)
    b2 = ax.bar(x,     per_f1,   w, label="F1 Score",  color='#10b981',
                edgecolor='white', linewidth=0.6, zorder=3)
    b3 = ax.bar(x + w, per_rec,  w, label="Recall",    color='#f59e0b',
                edgecolor='white', linewidth=0.6, zorder=3)

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha='center', va='bottom', fontsize=7.5)

    label_bars(b1); label_bars(b2); label_bars(b3)
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, fontsize=10)
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(f"Standalone — Per-Class Precision / F1 / Recall (Best Epoch {best_epoch})",
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "standalone_per_class_prec_rec.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[5/7] Precision/F1/Recall bar -> {out}")

    # ── 6. Confusion matrix heatmap ───────────────────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names,
                linewidths=0.5, ax=axes[0], annot_kws={"size": 10})
    axes[0].set_xlabel("Predicted Label", fontweight='bold', labelpad=8)
    axes[0].set_ylabel("True Label", fontweight='bold', labelpad=8)
    axes[0].set_title(f"Confusion Matrix — Raw Counts\n(Best Epoch {best_epoch})",
                      fontsize=11, fontweight='bold')

    # Normalised (recall per class)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=emotion_names, yticklabels=emotion_names,
                linewidths=0.5, ax=axes[1], annot_kws={"size": 10},
                vmin=0, vmax=1)
    axes[1].set_xlabel("Predicted Label", fontweight='bold', labelpad=8)
    axes[1].set_ylabel("True Label", fontweight='bold', labelpad=8)
    axes[1].set_title(f"Confusion Matrix — Normalised (Recall)\n(Best Epoch {best_epoch})",
                      fontsize=11, fontweight='bold')

    plt.suptitle("Standalone mini_XCEPTION — Confusion Analysis on FER-2013 Test Set",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "standalone_confusion_matrix.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[6/7] Confusion matrix        -> {out}")

    # ── 7. Research report (markdown) ────────────────────────────────────────
    rows = ""
    for i, ep in enumerate(epochs):
        rows += (f"| {ep} | {accuracy[i]:.4f} | {loss[i]:.4f} | {macro_f1[i]:.4f} |\n")

    per_class_table = ""
    for name, f1, p, r in zip(emotion_names, per_f1, per_prec, per_rec):
        per_class_table += f"| {name} | {p:.4f} | {r:.4f} | {f1:.4f} |\n"

    report = f"""# Standalone Emotion Classifier — Evaluation Report

**Architecture:** mini_XCEPTION (PyTorch port of oarriaga/face_classification)
**Dataset:** FER-2013 (Kaggle competition version)
**Training mode:** Centralised (no federated learning, no differential privacy)

---

## 1. Experimental Setup

| Parameter | Value |
|:---|:---|
| Architecture | mini_XCEPTION |
| Parameters | {n_params:,} |
| Training samples | {n_train:,} |
| Test samples | {n_test:,} |
| Epochs | {n_epochs} |
| Best epoch | {best_epoch} |
| Batch size | {bs} |
| Learning rate | {lr} |
| Optimiser | Adam + CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Augmentation | RandomHorizontalFlip, RandomAffine (rot=10, translate=0.1, scale=0.9-1.1) |
| Input resolution | 64 × 64 greyscale |

---

## 2. Best-Model Summary Metrics

| Metric | Value |
|:---|:---|
| **Validation Accuracy** | **{best_acc:.4f} ({best_acc:.2%})** |
| Macro F1 Score | {best_mf1:.4f} ({best_mf1:.2%}) |
| Weighted F1 Score | {best_wf1:.4f} ({best_wf1:.2%}) |
| Macro Precision | {best_prec:.4f} ({best_prec:.2%}) |
| Macro Recall | {best_rec:.4f} ({best_rec:.2%}) |

---

## 3. Per-Class Metrics (Best Checkpoint — Epoch {best_epoch})

| Emotion | Precision | Recall | F1 Score |
|:---|:---:|:---:|:---:|
{per_class_table}
**Macro avg** | {best_prec:.4f} | {best_rec:.4f} | {best_mf1:.4f} |

### sklearn Classification Report

```
{clf_report}
```

---

## 4. Epoch-by-Epoch Convergence

| Epoch | Val Accuracy | Val Loss | Macro F1 |
|:---:|:---:|:---:|:---:|
{rows}
---

## 5. Generated Figures

| Figure | Description |
|:---|:---|
| standalone_convergence.png | Validation accuracy & loss over 50 epochs |
| standalone_f1_convergence.png | Macro F1 curve with precision/recall reference lines |
| standalone_per_class_f1.png | Per-class F1 bar chart (best checkpoint) |
| standalone_per_class_prec_rec.png | Per-class Precision / F1 / Recall grouped bars |
| standalone_confusion_matrix.png | Raw-count + normalised confusion matrix (side by side) |

---

## 6. Comparison Context (vs Federated Learning)

The standalone model represents the **performance ceiling** for this architecture
on FER-2013 — trained with full data visibility and no privacy overhead.
The FL variant introduces:
- Non-IID data partitioning (Dirichlet α = 0.5) which reduces effective training signal
- Differential Privacy noise which perturbs gradients
- FedProx proximal regularisation which slows divergence but adds constraint

Any gap between FL and standalone accuracy quantifies the **privacy-utility trade-off**
and the **communication cost** of federated training.
"""

    rpt_path = os.path.join(OUT_DIR, "standalone_report.md")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[7/7] Research report         -> {rpt_path}")

    print("\n" + "=" * 60)
    print(f"  Best Accuracy   : {best_acc:.2%}  (Epoch {best_epoch})")
    print(f"  Macro F1        : {best_mf1:.4f}")
    print(f"  Weighted F1     : {best_wf1:.4f}")
    print(f"  Macro Precision : {best_prec:.4f}")
    print(f"  Macro Recall    : {best_rec:.4f}")
    print(f"  All outputs     : {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
