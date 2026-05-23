import flwr as fl
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import logging
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LBAAFedAvg")

class LBAAFedAvg(fl.server.strategy.FedAvg):
    """
    Layer-Based Anomaly Aware Federated Averaging (LBAAFedAvg).
    Acts as a secure aggregator that evaluates client parameter updates layer-by-layer
    to detect and discard anomalous updates (stealth poisoning attacks) before averaging.
    """
    def __init__(self, *args, anomaly_threshold=2.5, layer_flag_ratio=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomaly_threshold = anomaly_threshold  # Modified Z-score threshold
        self.layer_flag_ratio = layer_flag_ratio    # Ratio of anomalous layers to discard a client
        
        # Keep track of attack detection stats for the dashboard
        self.round_history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Overrides aggregate_fit to insert the Layer-Based Anomaly Aware filter."""
        if not results:
            return None, {}
            
        logger.info(f"--- Round {server_round}: Executing LBAAFedAvg Secure Filter ---")
        
        # 1. Deserialize parameters for each client
        client_updates = []
        client_num_samples = []
        client_cids = []
        client_attacker_flags = [] # If reported by the client for training tracking
        
        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(ndarrays)
            client_num_samples.append(fit_res.num_examples)
            client_cids.append(client.cid)
            
            # Retrieve optional attacker flag if present in metrics
            attacker = fit_res.metrics.get("attacker", 0)
            client_attacker_flags.append(bool(attacker))

        num_clients = len(client_updates)
        num_layers = len(client_updates[0])
        
        if num_clients <= 2:
            logger.warning("Insufficient clients for anomaly detection. Running standard FedAvg.")
            # Record empty attack metrics for this round
            self.round_history.append({
                "round": server_round,
                "total_clients": num_clients,
                "anomalous_clients": 0,
                "blocked_cids": [],
                "detected_attackers": []
            })
            return super().aggregate_fit(server_round, results, failures)

        # 2. Layer-by-Layer Anomaly Detection
        # anomalous_flags[client_idx][layer_idx] = True if anomalous
        anomalous_flags = np.zeros((num_clients, num_layers), dtype=bool)
        layer_distances = np.zeros((num_clients, num_layers))
        
        for l in range(num_layers):
            # Collect layer l parameters across all clients
            layer_params = [client_updates[k][l] for k in range(num_clients)]
            
            # Flatten each client's layer weights into a 1D vector
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
            
            # If MAD is virtually zero (all identical), avoid division by zero
            if mad < 1e-12:
                mad = 1e-12
                
            # Compute modified Z-scores
            # Modified Z-score = 0.6745 * (x - median) / MAD
            z_scores = 0.6745 * (distances - median_distance) / mad
            
            # Flag layer l of client k if its modified Z-score exceeds the threshold
            for k in range(num_clients):
                if z_scores[k] > self.anomaly_threshold:
                    anomalous_flags[k, l] = True

        # 3. Client Filtering based on anomalous layer ratio
        verified_results = []
        blocked_cids = []
        detected_attackers = [] # Log of verified ground-truth attackers caught
        
        # Calculate anomalous layer ratio for each client
        anomalous_layer_ratios = np.sum(anomalous_flags, axis=1) / num_layers
        
        for k in range(num_clients):
            flag_ratio = anomalous_layer_ratios[k]
            cid = client_cids[k]
            is_attacker = client_attacker_flags[k]
            
            logger.info(f"Client {cid} (Attacker Ground Truth: {is_attacker}) - "
                        f"Anomalous Layer Ratio: {flag_ratio:.2%} "
                        f"(Max Z-Score Distance: {np.max(layer_distances[k]):.4f})")
            
            if flag_ratio >= self.layer_flag_ratio:
                logger.warning(f"[WARNING] DISCARDED: Client {cid} detected as anomalous "
                               f"({flag_ratio:.2%} layers flagged). Update blocked.")
                blocked_cids.append(cid)
                if is_attacker:
                    detected_attackers.append(cid)
            else:
                verified_results.append(results[k])

        logger.info(f"LBAAFedAvg Filter Result: {len(verified_results)} verified, "
                    f"{len(blocked_cids)} blocked out of {num_clients} total updates.")

        # Save stats to strategy history
        self.round_history.append({
            "round": server_round,
            "total_clients": num_clients,
            "anomalous_clients": len(blocked_cids),
            "blocked_cids": blocked_cids,
            "detected_attackers": detected_attackers
        })

        # 4. Standard Weighted Aggregation on Verified Clients
        if not verified_results:
            logger.error("[ERROR] All client updates were blocked by LBAAFedAvg! Retaining current global model.")
            # Return current parameters without changes
            # Retrieve standard aggregated parameters from previous state or return None
            return None, {}
            
        # Call FedAvg aggregation on verified clients only
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, verified_results, failures
        )
        
        # Include defense stats in metrics returned to simulation
        aggregated_metrics["total_clients"] = num_clients
        aggregated_metrics["anomalous_clients"] = len(blocked_cids)
        aggregated_metrics["blocked_cids_count"] = len(blocked_cids)
        
        return aggregated_parameters, aggregated_metrics
