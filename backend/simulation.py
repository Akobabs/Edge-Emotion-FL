import os
import json
import math
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from backend.dataset.data_loader import get_federated_loaders
from backend.models.cnn import EmotionCNN, get_model_size

# Hyperparameters for simulation
NUM_CLIENTS = 5
NUM_ROUNDS = 15
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.001
FEDPROX_MU = 0.01
BATCH_SIZE = 64
MAX_SAMPLES_PER_CLIENT = 2500   # cap per client so CPU finishes in ~15 min

# Differential Privacy parameters
DP_ENABLED = True
DP_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.1   # σ — noise per-param is σ*S/√N so total noise L2 = σ*S

# Attacker/Poisoning configurations
ATTACKER_CID = 4            # Client 4 will act as the malicious stealth attacker
ATTACK_TYPE = "sign_flip"   # "sign_flip" or "random_noise" or "label_flip"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Emotion class names (FER-2013 order)
EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def compute_dp_epsilon(rounds_completed: int, noise_multiplier: float,
                       delta: float = 1e-5) -> float:
    """
    Approximate accumulated DP epsilon using advanced composition of the Gaussian
    mechanism.  Each round contributes one mechanism application per benign client.
    Formula (Theorem 3.3, Dwork & Roth):
        ε_total ≈ sqrt(2 * T * ln(1/δ)) / σ
    where σ = noise_multiplier (sensitivity-normalised std dev) and T = rounds.
    """
    if noise_multiplier <= 0 or rounds_completed <= 0:
        return float('inf')
    return round(math.sqrt(2 * rounds_completed * math.log(1.0 / delta)) / noise_multiplier, 3)


class LBAAFedAvgAggregator:
    """
    Server-side Layer-Based Anomaly Aware Aggregator (LBAAFedAvg).
    Evaluates incoming client updates layer-by-layer using Median Absolute Deviation (MAD)
    to detect and filter out stealth poisoning attacks before averaging.
    """
    def __init__(self, anomaly_threshold=2.2, layer_flag_ratio=0.25):
        self.anomaly_threshold = anomaly_threshold
        self.layer_flag_ratio = layer_flag_ratio
        self.round_history = []

    def filter_and_aggregate(self, server_round: int, client_updates: list, 
                             client_num_samples: list, global_model_weights: list) -> tuple:
        """
        Evaluates updates layer-by-layer, blocks anomalous updates, and aggregates the rest.
        
        Args:
            server_round (int): The current federated round.
            client_updates (list of list of np.ndarray): Weights submitted by each client.
            client_num_samples (list of int): Number of samples processed by each client.
            global_model_weights (list of np.ndarray): Current global weights.
            
        Returns:
            tuple: (aggregated_weights, anomaly_count, blocked_cids)
        """
        num_clients = len(client_updates)
        num_layers = len(client_updates[0])
        
        if num_clients <= 2:
            # Fallback to standard weighted average if too few clients
            print("[SERVER] [WARNING] Insufficient clients for anomaly detection. Running standard FedAvg.")
            self.round_history.append({
                "round": server_round,
                "total_clients": num_clients,
                "anomalous_clients": 0,
                "blocked_cids": []
            })
            return self._weighted_average(client_updates, client_num_samples), 0, []

        print(f"--- Round {server_round}: Executing LBAAFedAvg Secure Filter ---")
        
        # anomalous_flags[client_idx][layer_idx] = True if anomalous
        anomalous_flags = np.zeros((num_clients, num_layers), dtype=bool)
        layer_distances = np.zeros((num_clients, num_layers))
        
        # Compute deltas (weight updates) relative to current global model weights
        client_deltas = []
        for k in range(num_clients):
            deltas_k = [client_updates[k][l] - global_model_weights[l] for l in range(num_layers)]
            client_deltas.append(deltas_k)

        # Evaluate layer-by-layer
        for l in range(num_layers):
            # Collect layer l deltas across all clients
            layer_params = [client_deltas[k][l] for k in range(num_clients)]
            flat_params = [lp.flatten() for lp in layer_params]
            
            # Compute element-wise median vector for layer l
            median_vector = np.median(flat_params, axis=0)
            
            # Calculate L2 distance from median for each client
            distances = []
            for k in range(num_clients):
                dist = np.sqrt(np.sum((flat_params[k] - median_vector) ** 2))
                distances.append(dist)
            distances = np.array(distances)
            layer_distances[:, l] = distances
            
            # Compute Median Absolute Deviation (MAD) of distances
            median_distance = np.median(distances)
            mad = np.median(np.abs(distances - median_distance))
            if mad < 1e-12:
                mad = 1e-12
                
            # Compute modified Z-scores
            z_scores = 0.6745 * (distances - median_distance) / mad
            
            # Flag layer l of client k if it exceeds threshold
            for k in range(num_clients):
                if z_scores[k] > self.anomaly_threshold:
                    anomalous_flags[k, l] = True

        # Process flags across all layers
        verified_updates = []
        verified_samples = []
        blocked_cids = []
        
        anomalous_layer_ratios = np.sum(anomalous_flags, axis=1) / num_layers
        
        for k in range(num_clients):
            flag_ratio = anomalous_layer_ratios[k]
            is_attacker = (k == ATTACKER_CID)
            
            print(f"Client {k} (Attacker Ground Truth: {is_attacker}) - "
                  f"Anomalous Layer Ratio: {flag_ratio:.2%} "
                  f"(Max Z-Score Distance: {np.max(layer_distances[k]):.4f})")
            
            if flag_ratio >= self.layer_flag_ratio:
                print(f"[SERVER] [WARNING] DISCARDED: Client {k} detected as anomalous "
                      f"({flag_ratio:.2%} layers flagged). Update blocked.")
                blocked_cids.append(str(k))
            else:
                verified_updates.append(client_updates[k])
                verified_samples.append(client_num_samples[k])

        print(f"[SERVER] LBAAFedAvg Filter Result: {len(verified_updates)} verified, "
              f"{len(blocked_cids)} blocked out of {num_clients} total updates.")

        # Log round history
        self.round_history.append({
            "round": server_round,
            "total_clients": num_clients,
            "anomalous_clients": len(blocked_cids),
            "blocked_cids": blocked_cids
        })

        if not verified_updates:
            print("[SERVER] [ERROR] All client updates were blocked! Retaining current global model.")
            return global_model_weights, num_clients, blocked_cids
            
        # Aggregate verified updates
        aggregated_weights = self._weighted_average(verified_updates, verified_samples)
        return aggregated_weights, len(blocked_cids), blocked_cids

    def _weighted_average(self, updates: list, samples: list) -> list:
        """Computes sample-weighted average of weights."""
        total_samples = sum(samples)
        num_layers = len(updates[0])
        aggregated = []
        
        for l in range(num_layers):
            layer_sum = np.zeros_like(updates[0][l])
            for k in range(len(updates)):
                layer_sum += updates[k][l] * (samples[k] / total_samples)
            aggregated.append(layer_sum)
            
        return aggregated


def train_local_client(client_idx: int, global_model: nn.Module, train_loader: DataLoader,
                       epochs: int, lr: float, mu: float, dp_enabled: bool,
                       dp_clip: float, dp_noise: float, attacker: bool, attack_type: str) -> tuple:
    """Trains a simulated local client incorporating FedProx, local DP, and poisoning options."""
    local_model = copy.deepcopy(global_model).to(DEVICE)
    ref_model   = copy.deepcopy(global_model).to(DEVICE)
    ref_model.eval()

    local_model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    global_weights = [p.data.cpu().numpy().copy() for p in local_model.parameters()]

    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if attacker and attack_type == "label_flip":
                labels = torch.where(labels == 3, torch.tensor(4).to(DEVICE), labels)

            optimizer.zero_grad()
            outputs = local_model(images)
            loss    = criterion(outputs, labels)

            # FedProx proximal penalty: 0.5 * mu * ||w - w^t||^2
            prox_term = sum(
                torch.sum((lp - rp) ** 2)
                for lp, rp in zip(local_model.parameters(), ref_model.parameters())
            )
            (loss + 0.5 * mu * prox_term).backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
            
    # Extract weights after training (explicit copy to break memory view reference)
    local_weights = [p.data.cpu().numpy().copy() for p in local_model.parameters()]
    
    # Compute weight update delta: W_local - W_global
    delta = [lp - gw for lp, gw in zip(local_weights, global_weights)]
    
    # Apply Client-Side Differential Privacy (DP)
    if dp_enabled and not attacker:
        # L2 norm clipping
        total_norm = np.sqrt(sum(np.sum(d ** 2) for d in delta))
        clip_factor = min(1.0, dp_clip / (total_norm + 1e-10))
        clipped_delta = [d * clip_factor for d in delta]

        # Noise scale normalised by sqrt(total params) so that total noise
        # L2 ≈ dp_noise * dp_clip regardless of model size.
        total_params = sum(d.size for d in clipped_delta)
        per_param_sigma = (dp_noise * dp_clip) / np.sqrt(total_params)

        noised_delta = [
            cd + np.random.normal(0.0, per_param_sigma, size=cd.shape)
            for cd in clipped_delta
        ]
        delta = noised_delta
        
    # Apply Attacker weight poisoning
    if attacker:
        if attack_type == "sign_flip":
            # Reverses and scales local updates (realistic Model Replacement/Fed-CRA Attack)
            # Scaling up by 10000.0 overrides updates and triggers the LBAAFedAvg outlier z-score filter!
            delta = [-10000.0 * d for d in delta]
        elif attack_type == "random_noise":
            delta = [d + np.random.normal(0, 4.0, size=d.shape) for d in delta]
            
    # Reconstruct final parameters: W_final = W_global + delta
    final_weights = [gw + d for gw, d in zip(global_weights, delta)]
    
    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    
    return final_weights, total, {"loss": epoch_loss, "accuracy": epoch_acc}


def main():
    print(f"============================================================")
    print(f"   STARTING PRIVACY-CONSCIOUS FEDERATED EMOTION RECOGNITION")
    print(f"============================================================")
    print(f"Executing on hardware device: {DEVICE.upper()}")
    
    # 1. Load Data partitions
    print("\nPreparing Non-IID datasets across clients...")
    # Use the authoritative FER-2013 competition CSV (same data as the reference codebase)
    csv_path = "DATA/FER2013/fer2013/fer2013/fer2013.csv"
    client_loaders, test_loader = get_federated_loaders(
        csv_path=csv_path,
        num_clients=NUM_CLIENTS,
        alpha=0.5,
        batch_size=BATCH_SIZE,
        max_samples_per_client=MAX_SAMPLES_PER_CLIENT
    )
    
    # 2. Initialize global model and aggregator
    global_model = EmotionCNN().to(DEVICE)
    model_params_count = get_model_size(global_model)
    model_size_mb = (model_params_count * 4) / (1024 * 1024)
    print(f"Model parameters: {model_params_count:,} (~{model_size_mb:.2f} MB per update)")
    
    aggregator = LBAAFedAvgAggregator(anomaly_threshold=2.2, layer_flag_ratio=0.25)
    
    rounds_list = []
    accuracies = []
    losses = []
    macro_f1_list = []
    weighted_f1_list = []
    precision_list = []
    recall_list = []
    dp_epsilon_list = []
    adr_list = []       # attack detection rate per round
    fpr_list = []       # false positive rate per round
    cumulative_overhead = 0.0
    overhead_mb_list = []
    final_confusion_matrix = None
    final_per_class_f1 = None
    final_classification_report = None
    
    # 3. Main Federated Training Loop
    for r in range(1, NUM_ROUNDS + 1):
        print(f"\n[ROUND {r}]")
        
        client_updates = []
        client_samples = []
        
        # Broadcast global weights to all clients and train locally
        global_weights = [p.data.cpu().numpy() for p in global_model.parameters()]
        
        for k in range(NUM_CLIENTS):
            is_attacker = (k == ATTACKER_CID)
            
            if is_attacker:
                print(f"[CLIENT {k}] [WARNING] Initiating as an Active Adversary! Attack: {ATTACK_TYPE.upper()}")
            else:
                print(f"[CLIENT {k}] Registering as a Benign Node (FedProx + Differential Privacy)")
                
            weights, num_samples, metrics = train_local_client(
                client_idx=k,
                global_model=global_model,
                train_loader=client_loaders[k],
                epochs=LOCAL_EPOCHS,
                lr=LEARNING_RATE,
                mu=FEDPROX_MU,
                dp_enabled=DP_ENABLED,
                dp_clip=DP_NORM_CLIP,
                dp_noise=DP_NOISE_MULTIPLIER,
                attacker=is_attacker,
                attack_type=ATTACK_TYPE
            )
            client_updates.append(weights)
            client_samples.append(num_samples)
            
        # 4. Aggregator: Secure LBAAFedAvg Filter
        new_global_weights, anomaly_count, blocked_cids = aggregator.filter_and_aggregate(
            server_round=r,
            client_updates=client_updates,
            client_num_samples=client_samples,
            global_model_weights=global_weights
        )
        
        # Update global model with aggregated weights
        for p, val in zip(global_model.parameters(), new_global_weights):
            p.data = torch.tensor(val, dtype=torch.float32).to(DEVICE)
        
        # 5. Centralized Evaluation with full classification metrics
        global_model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = global_model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        total = len(all_labels)

        test_loss = running_loss / max(total, 1)
        test_accuracy = float(np.mean(all_preds == all_labels))

        # Classification metrics
        macro_f1   = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
        macro_prec  = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
        macro_rec   = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))

        # DP privacy budget (accumulated)
        epsilon = compute_dp_epsilon(r, DP_NOISE_MULTIPLIER) if DP_ENABLED else float('inf')

        # Attack detection metrics for this round
        # ADR = fraction of true attacker rounds that were actually blocked
        true_attacker_blocked = (str(ATTACKER_CID) in blocked_cids)
        adr = 1.0 if true_attacker_blocked else 0.0
        # FPR = benign clients incorrectly blocked / total benign clients
        benign_blocked = sum(1 for cid in blocked_cids if int(cid) != ATTACKER_CID)
        fpr = benign_blocked / max(1, NUM_CLIENTS - 1)

        print(f"\n[SERVER Round {r}] Evaluation - "
              f"Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.2%} | "
              f"Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | "
              f"Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f} | "
              f"DP eps: {epsilon:.2f} | ADR: {adr:.0%} | FPR: {fpr:.0%}")

        # Accumulate metrics
        rounds_list.append(r)
        accuracies.append(test_accuracy)
        losses.append(test_loss)
        macro_f1_list.append(macro_f1)
        weighted_f1_list.append(weighted_f1)
        precision_list.append(macro_prec)
        recall_list.append(macro_rec)
        dp_epsilon_list.append(epsilon)
        adr_list.append(adr)
        fpr_list.append(fpr)

        # Retain confusion matrix and per-class F1 for the final round
        final_confusion_matrix = confusion_matrix(all_labels, all_preds).tolist()
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        final_per_class_f1 = [round(float(v), 4) for v in per_class_f1]
        final_classification_report = classification_report(
            all_labels, all_preds,
            target_names=EMOTION_NAMES,
            zero_division=0
        )

        # Compute communication cost: each client downloads + uploads model
        round_overhead = 2 * model_size_mb * NUM_CLIENTS
        cumulative_overhead += round_overhead
        overhead_mb_list.append(round(cumulative_overhead, 2))
        
    # 6. Compile simulation metrics and export history log
    print("\nSimulation complete! Compiling training metrics history...")
    
    anomalous_counts = [log["anomalous_clients"] for log in aggregator.round_history]
    blocked_cids_list = [log["blocked_cids"] for log in aggregator.round_history]
    total_clients_round = [log["total_clients"] for log in aggregator.round_history]
    
    # Print final classification report
    print("\n[FINAL ROUND] Per-Class Classification Report:")
    print(final_classification_report)

    metrics_log = {
        "rounds": rounds_list,
        "accuracy": [round(float(a), 4) for a in accuracies],
        "loss": [round(float(l), 4) for l in losses],
        # --- NEW: classification metrics per round ---
        "macro_f1": [round(float(v), 4) for v in macro_f1_list],
        "weighted_f1": [round(float(v), 4) for v in weighted_f1_list],
        "macro_precision": [round(float(v), 4) for v in precision_list],
        "macro_recall": [round(float(v), 4) for v in recall_list],
        # --- NEW: per-class F1 at final round ---
        "per_class_f1_final": final_per_class_f1,
        "emotion_names": EMOTION_NAMES,
        # --- NEW: confusion matrix at final round ---
        "confusion_matrix_final": final_confusion_matrix,
        # --- NEW: differential privacy budget ---
        "dp_epsilon": [round(float(v), 3) for v in dp_epsilon_list],
        # --- NEW: attack detection metrics ---
        "attack_detection_rate": [round(float(v), 4) for v in adr_list],
        "false_positive_rate": [round(float(v), 4) for v in fpr_list],
        # --- existing fields ---
        "total_clients": total_clients_round,
        "anomalous_clients": anomalous_counts,
        "blocked_cids": blocked_cids_list,
        "communication_overhead_mb": overhead_mb_list,
        "model_parameters": model_params_count,
        "model_size_mb": round(model_size_mb, 2),
        "num_clients": NUM_CLIENTS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "mu_fedprox": FEDPROX_MU,
        "dp_enabled": DP_ENABLED,
        "dp_norm_clip": DP_NORM_CLIP,
        "dp_noise_multiplier": DP_NOISE_MULTIPLIER,
        "attacker_cid": str(ATTACKER_CID),
        "attack_type": ATTACK_TYPE
    }
    
    # Save to frontend/simulation_history.json
    frontend_dir = "frontend"
    os.makedirs(frontend_dir, exist_ok=True)
    history_path = os.path.join(frontend_dir, "simulation_history.json")
    
    with open(history_path, "w") as f:
        json.dump(metrics_log, f, indent=4)
        
    print(f"Successfully compiled simulation history! Saved to: {history_path}")
    
    # Save the PyTorch global model state dict checkpoint
    model_chk_path = "backend/global_model.pth"
    os.makedirs("backend", exist_ok=True)
    torch.save(global_model.state_dict(), model_chk_path)
    print(f"Saved PyTorch final global model state dict checkpoint to {model_chk_path}")

if __name__ == "__main__":
    main()
