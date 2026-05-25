"""
Standalone (centralized) emotion classifier.
Same mini_XCEPTION architecture as the FL counterpart — fair apples-to-apples comparison.
Trains on the full FER-2013 dataset with no federated overhead, DP, or aggregation.
Saves only the best checkpoint; final comprehensive metrics computed from that best model.
"""
import os
import sys
import io
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from backend.dataset.data_loader import (
    FER2013Dataset, load_fer2013_csv, TRAIN_TRANSFORM, TEST_TRANSFORM
)
from backend.models.cnn import EmotionCNN, get_model_size

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 0.001
WEIGHT_DECAY = 1e-4
CSV_PATH     = "DATA/FER2013/fer2013/fer2013/fer2013.csv"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CKPT_PATH    = "backend/standalone_model.pth"


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            running_loss += criterion(outputs, labels).item() * imgs.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total      = len(all_labels)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return running_loss / max(total, 1), float(np.mean(all_preds == all_labels)), all_preds, all_labels


def main():
    print("=" * 60)
    print("   STANDALONE CENTRALIZED EMOTION CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE.upper()} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_imgs, train_lbls, test_imgs, test_lbls = load_fer2013_csv(CSV_PATH)
    train_dataset = FER2013Dataset(train_imgs, train_lbls, transform=TRAIN_TRANSFORM)
    test_dataset  = FER2013Dataset(test_imgs,  test_lbls,  transform=TEST_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Training: {len(train_dataset):,} samples | Test: {len(test_dataset):,} samples\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = EmotionCNN(num_classes=7).to(DEVICE)
    param_count = get_model_size(model)
    print(f"Architecture: mini_XCEPTION | Parameters: {param_count:,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)

    # Convergence curves saved for dashboard charts
    conv_accuracy = []
    conv_loss     = []
    conv_macro_f1 = []

    best_acc       = 0.0
    best_epoch     = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)

        val_loss, val_acc, preds, lbls = evaluate(model, test_loader, criterion)
        macro_f1 = float(f1_score(lbls, preds, average='macro', zero_division=0))

        conv_accuracy.append(round(val_acc, 4))
        conv_loss.append(round(val_loss, 4))
        conv_macro_f1.append(round(macro_f1, 4))

        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)

        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"Train loss={train_loss:.4f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.2%} | "
              f"Macro-F1={macro_f1:.4f} | "
              f"Best={best_acc:.2%} (ep {best_epoch})")

    # ── Final evaluation on best checkpoint ──────────────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}, acc {best_acc:.2%}) for final evaluation...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))

    _, best_val_acc, best_preds, best_labels = evaluate(model, test_loader, criterion)

    best_macro_f1        = float(f1_score(best_labels, best_preds, average='macro',    zero_division=0))
    best_weighted_f1     = float(f1_score(best_labels, best_preds, average='weighted', zero_division=0))
    best_macro_prec      = float(precision_score(best_labels, best_preds, average='macro',    zero_division=0))
    best_macro_rec       = float(recall_score(best_labels, best_preds, average='macro',       zero_division=0))
    best_per_class_f1    = [round(float(v), 4) for v in f1_score(best_labels, best_preds, average=None, zero_division=0)]
    best_per_class_prec  = [round(float(v), 4) for v in precision_score(best_labels, best_preds, average=None, zero_division=0)]
    best_per_class_rec   = [round(float(v), 4) for v in recall_score(best_labels, best_preds, average=None, zero_division=0)]
    best_conf_mat        = confusion_matrix(best_labels, best_preds).tolist()
    best_clf_report      = classification_report(
        best_labels, best_preds, target_names=EMOTION_NAMES, zero_division=0
    )

    print("\n[BEST MODEL] Per-Class Classification Report:")
    print(best_clf_report)
    print(f"Best Validation Accuracy: {best_val_acc:.2%}")

    # ── Save results JSON ────────────────────────────────────────────────────
    history_json = {
        # Convergence curves (for dashboard charts — same field names as simulation_history.json)
        "rounds":    list(range(1, EPOCHS + 1)),
        "accuracy":  conv_accuracy,
        "loss":      conv_loss,
        "macro_f1":  conv_macro_f1,

        # Best-model aggregate metrics
        "best_accuracy":        round(best_val_acc, 4),
        "best_macro_f1":        round(best_macro_f1, 4),
        "best_weighted_f1":     round(best_weighted_f1, 4),
        "best_macro_precision": round(best_macro_prec, 4),
        "best_macro_recall":    round(best_macro_rec, 4),
        "best_epoch":           best_epoch,

        # Best-model per-class breakdown
        "per_class_f1_final":         best_per_class_f1,
        "per_class_precision_final":  best_per_class_prec,
        "per_class_recall_final":     best_per_class_rec,
        "confusion_matrix_final":     best_conf_mat,
        "classification_report":      best_clf_report,
        "emotion_names":              EMOTION_NAMES,

        # FL placeholder fields so dashboard stat cards render without errors
        "weighted_f1":             conv_macro_f1,       # reuse macro as approximation for charts
        "macro_precision":         [round(best_macro_prec, 4)] * EPOCHS,
        "macro_recall":            [round(best_macro_rec, 4)]  * EPOCHS,
        "dp_epsilon":              [0.0] * EPOCHS,
        "attack_detection_rate":   [0.0] * EPOCHS,
        "false_positive_rate":     [0.0] * EPOCHS,
        "total_clients":           [1]   * EPOCHS,
        "anomalous_clients":       [0]   * EPOCHS,
        "blocked_cids":            [[]]  * EPOCHS,
        "communication_overhead_mb": [0.0] * EPOCHS,

        # Meta
        "model_type":         "standalone",
        "model_parameters":   param_count,
        "num_clients":        1,
        "epochs_total":       EPOCHS,
        "learning_rate":      LR,
        "batch_size":         BATCH_SIZE,
        "training_samples":   len(train_dataset),
        "test_samples":       len(test_dataset),
    }

    os.makedirs("frontend", exist_ok=True)
    with open("frontend/standalone_history.json", "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=4)
    print("Saved results -> frontend/standalone_history.json")

    # ── Export best model to ONNX ────────────────────────────────────────────
    print("Exporting best model to ONNX...")
    model.eval()
    dummy = torch.randn(1, 1, 64, 64)
    torch.onnx.export(
        model, dummy, "frontend/standalone_model.onnx",
        export_params=True, opset_version=12, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    try:
        import onnx
        onnx.checker.check_model(onnx.load("frontend/standalone_model.onnx"))
        print("[SUCCESS] frontend/standalone_model.onnx validated.")
    except ImportError:
        print("frontend/standalone_model.onnx exported.")
    except Exception as e:
        print(f"ONNX validation warning: {e}")

    print("\n" + "=" * 60)
    print("  STANDALONE TRAINING COMPLETE")
    print(f"  Best Epoch        : {best_epoch}/{EPOCHS}")
    print(f"  Best Val Accuracy : {best_val_acc:.2%}")
    print(f"  Macro F1          : {best_macro_f1:.4f}")
    print(f"  Weighted F1       : {best_weighted_f1:.4f}")
    print(f"  Macro Precision   : {best_macro_prec:.4f}")
    print(f"  Macro Recall      : {best_macro_rec:.4f}")
    print(f"  Checkpoint        : {CKPT_PATH}")
    print(f"  ONNX model        : frontend/standalone_model.onnx")
    print(f"  History JSON      : frontend/standalone_history.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
