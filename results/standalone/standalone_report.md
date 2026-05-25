# Standalone Emotion Classifier — Evaluation Report

**Architecture:** mini_XCEPTION (PyTorch port of oarriaga/face_classification)
**Dataset:** FER-2013 (Kaggle competition version)
**Training mode:** Centralised (no federated learning, no differential privacy)

---

## 1. Experimental Setup

| Parameter | Value |
|:---|:---|
| Architecture | mini_XCEPTION |
| Parameters | 56,951 |
| Training samples | 28,709 |
| Test samples | 7,178 |
| Epochs | 50 |
| Best epoch | 47 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimiser | Adam + CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Augmentation | RandomHorizontalFlip, RandomAffine (rot=10, translate=0.1, scale=0.9-1.1) |
| Input resolution | 64 × 64 greyscale |

---

## 2. Best-Model Summary Metrics

| Metric | Value |
|:---|:---|
| **Validation Accuracy** | **0.6512 (65.12%)** |
| Macro F1 Score | 0.6052 (60.52%) |
| Weighted F1 Score | 0.6434 (64.34%) |
| Macro Precision | 0.6392 (63.92%) |
| Macro Recall | 0.5942 (59.42%) |

---

## 3. Per-Class Metrics (Best Checkpoint — Epoch 47)

| Emotion | Precision | Recall | F1 Score |
|:---|:---:|:---:|:---:|
| Angry | 0.5688 | 0.5866 | 0.5776 |
| Disgust | 0.7037 | 0.3423 | 0.4606 |
| Fear | 0.5151 | 0.3828 | 0.4392 |
| Happy | 0.8243 | 0.8884 | 0.8551 |
| Sad | 0.5385 | 0.4820 | 0.5087 |
| Surprise | 0.7440 | 0.7870 | 0.7649 |
| Neutral | 0.5797 | 0.6902 | 0.6301 |

**Macro avg** | 0.6392 | 0.5942 | 0.6052 |

### sklearn Classification Report

```
              precision    recall  f1-score   support

       Angry       0.57      0.59      0.58       958
     Disgust       0.70      0.34      0.46       111
        Fear       0.52      0.38      0.44      1024
       Happy       0.82      0.89      0.86      1774
         Sad       0.54      0.48      0.51      1247
    Surprise       0.74      0.79      0.76       831
     Neutral       0.58      0.69      0.63      1233

    accuracy                           0.65      7178
   macro avg       0.64      0.59      0.61      7178
weighted avg       0.64      0.65      0.64      7178

```

---

## 4. Epoch-by-Epoch Convergence

| Epoch | Val Accuracy | Val Loss | Macro F1 |
|:---:|:---:|:---:|:---:|
| 1 | 0.4021 | 1.6287 | 0.2623 |
| 2 | 0.4531 | 1.5578 | 0.3898 |
| 3 | 0.5194 | 1.4180 | 0.4205 |
| 4 | 0.5458 | 1.3639 | 0.4440 |
| 5 | 0.5521 | 1.3587 | 0.4685 |
| 6 | 0.5453 | 1.3639 | 0.4420 |
| 7 | 0.5666 | 1.3127 | 0.4863 |
| 8 | 0.5695 | 1.3274 | 0.4687 |
| 9 | 0.5777 | 1.2834 | 0.5020 |
| 10 | 0.5936 | 1.2761 | 0.5192 |
| 11 | 0.5747 | 1.3019 | 0.5070 |
| 12 | 0.5933 | 1.2787 | 0.5198 |
| 13 | 0.5991 | 1.2600 | 0.5293 |
| 14 | 0.5978 | 1.2527 | 0.5371 |
| 15 | 0.6024 | 1.2580 | 0.5290 |
| 16 | 0.6014 | 1.2530 | 0.5458 |
| 17 | 0.6087 | 1.2382 | 0.5547 |
| 18 | 0.6062 | 1.2420 | 0.5507 |
| 19 | 0.6126 | 1.2490 | 0.5403 |
| 20 | 0.6094 | 1.2621 | 0.5368 |
| 21 | 0.6201 | 1.2329 | 0.5636 |
| 22 | 0.6201 | 1.2360 | 0.5595 |
| 23 | 0.6034 | 1.2707 | 0.5527 |
| 24 | 0.6244 | 1.2156 | 0.5643 |
| 25 | 0.6208 | 1.2396 | 0.5641 |
| 26 | 0.6276 | 1.2213 | 0.5694 |
| 27 | 0.6255 | 1.2260 | 0.5719 |
| 28 | 0.6337 | 1.2075 | 0.5730 |
| 29 | 0.6356 | 1.2109 | 0.5844 |
| 30 | 0.6257 | 1.2174 | 0.5745 |
| 31 | 0.6388 | 1.1990 | 0.5921 |
| 32 | 0.6381 | 1.1965 | 0.5892 |
| 33 | 0.6418 | 1.1909 | 0.5974 |
| 34 | 0.6396 | 1.2037 | 0.5855 |
| 35 | 0.6361 | 1.1998 | 0.5910 |
| 36 | 0.6413 | 1.1877 | 0.5937 |
| 37 | 0.6362 | 1.1954 | 0.5866 |
| 38 | 0.6446 | 1.1880 | 0.6031 |
| 39 | 0.6402 | 1.1979 | 0.5937 |
| 40 | 0.6457 | 1.1813 | 0.6008 |
| 41 | 0.6454 | 1.1849 | 0.6093 |
| 42 | 0.6474 | 1.1859 | 0.6074 |
| 43 | 0.6489 | 1.1845 | 0.6012 |
| 44 | 0.6450 | 1.1820 | 0.5998 |
| 45 | 0.6463 | 1.1785 | 0.6005 |
| 46 | 0.6478 | 1.1791 | 0.6054 |
| 47 | 0.6512 | 1.1821 | 0.6052 |
| 48 | 0.6509 | 1.1806 | 0.6073 |
| 49 | 0.6496 | 1.1773 | 0.6056 |
| 50 | 0.6481 | 1.1808 | 0.6047 |

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
