import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import flwr as fl
from collections import OrderedDict
from backend.models.cnn import EmotionCNN

class FedProxClient(fl.client.NumPyClient):
    """
    Custom Flower Client implementing:
    - FedProx proximal regularization (handles statistical data heterogeneity).
    - Local Differential Privacy (clipping and noise injection for parameter updates).
    - Optional Poisoning Attack Simulation (for validating the LBAAFedAvg server defense).
    """
    def __init__(self, client_id, train_loader, test_loader, device="cpu", 
                 mu=0.1, dp_enabled=True, dp_norm_clip=1.0, dp_noise_multiplier=0.05,
                 attacker=False, attack_type="sign_flip"):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.mu = mu
        
        # Differential Privacy parameters
        self.dp_enabled = dp_enabled
        self.dp_norm_clip = dp_norm_clip
        self.dp_noise_multiplier = dp_noise_multiplier
        
        # Attacker simulation parameters
        self.attacker = attacker
        self.attack_type = attack_type # "sign_flip", "random_noise", or "label_flip"
        
        # Initialize local CNN model
        self.model = EmotionCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        """Returns local model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Loads weights into the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Executes local training on the client's partition.
        Implements FedProx regularization and Differential Privacy update noise.
        """
        # Save current global weights for FedProx and Differential Privacy
        self.set_parameters(parameters)
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        
        # Set model to training mode
        self.model.train()
        
        # Retrieve hyperparams from config or fall back to defaults
        epochs = int(config.get("local_epochs", 2))
        lr = float(config.get("lr", 0.01))
        
        # Use Stochastic Gradient Descent (SGD)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Execute local training loop
        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Attacker Simulation: Label Flip (e.g. flip class 3 [Happy] to class 4 [Sad])
                if self.attacker and self.attack_type == "label_flip":
                    labels = torch.where(labels == 3, torch.tensor(4).to(self.device), labels)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # Standard Categorical Cross-Entropy Loss
                loss = self.criterion(outputs, labels)
                
                # Apply FedProx Proximal Regularization Term
                # 0.5 * mu * || w - w^t ||_2^2
                prox_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_model.parameters()):
                    prox_term += torch.sum((local_param - global_param) ** 2)
                
                total_loss = loss + 0.5 * self.mu * prox_term
                
                # Backpropagation
                total_loss.backward()
                optimizer.step()
                
                # Accumulate metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Extract weights after training
        local_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        # Compute parameter update delta: W_local - W_global
        delta = [lp - gp for lp, gp in zip(local_params, parameters)]
        
        # Apply Client-Side Differential Privacy (DP)
        if self.dp_enabled and not self.attacker:
            # 1. Compute total L2 norm of the update across all layers
            total_norm = np.sqrt(sum(np.sum(d ** 2) for d in delta))
            
            # 2. L2 Norm Clipping
            clip_factor = min(1.0, self.dp_norm_clip / (total_norm + 1e-10))
            clipped_delta = [d * clip_factor for d in delta]
            
            # 3. Add Calibrated Gaussian Noise: N(0, sigma^2 * S^2)
            noised_delta = []
            for cd in clipped_delta:
                noise = np.random.normal(
                    loc=0.0, 
                    scale=self.dp_noise_multiplier * self.dp_norm_clip, 
                    size=cd.shape
                )
                noised_delta.append(cd + noise)
                
            delta = noised_delta
            
        # Attacker Simulation: Weight-based attacks (stealth poisoning)
        if self.attacker:
            if self.attack_type == "sign_flip":
                # Sign flip: reverse update direction to degrade model convergence
                delta = [-2.0 * d for d in delta]
            elif self.attack_type == "random_noise":
                # Inject huge random noise updates to disrupt the server state
                delta = [d + np.random.normal(0, 5.0, size=d.shape) for d in delta]
                
        # Reconstruct final parameters: W_final = W_global + delta
        final_params = [gp + d for gp, d in zip(parameters, delta)]
        
        # Return parameters, sample count, and training performance
        return final_params, total, {
            "loss": epoch_loss, 
            "accuracy": epoch_acc, 
            "attacker": int(self.attacker)
        }

    def evaluate(self, parameters, config):
        """Evaluates model performance on the local validation split."""
        self.set_parameters(parameters)
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        
        return val_loss, total, {"accuracy": val_acc}
