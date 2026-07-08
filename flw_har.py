import flwr as fl
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

# ====================== PROTOTYPE REGULARIZATION (MFCPL) ======================
class PrototypeRegularization:
    """
    Based on MFCPL (Le et al., 2024)
    Cross-Modal Prototype Contrastive Learning for missing modality handling
    """
    def __init__(self, num_classes, embedding_dim, temperature=0.1):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.global_prototypes = torch.zeros(num_classes, embedding_dim)
        self.prototype_counts = torch.zeros(num_classes)
        
    def update_global_prototypes(self, client_features, client_labels):
        """
        Update global prototypes using client features
        Eq. 4 from MFCPL: P^k = (1/N) * sum(p_{Ω_i}^k)
        """
        for class_id in range(self.num_classes):
            class_features = client_features[client_labels == class_id]
            if len(class_features) > 0:
                local_prototype = class_features.mean(dim=0)
                self.global_prototypes[class_id] = (
                    self.global_prototypes[class_id] * self.prototype_counts[class_id] + 
                    local_prototype
                ) / (self.prototype_counts[class_id] + 1)
                self.prototype_counts[class_id] += 1
                
    def prototype_contrastive_loss(self, features, labels):
        """
        Cross-Modal Prototypes Contrastive (CMPC) Loss
        Eq. 8 from MFCPL: Encourages modality-specific representations to align with prototypes
        """
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.global_prototypes, dim=1)
        
        loss = 0.0
        for i in range(batch_size):
            label = labels[i]
            positive_prototype = prototypes[label].unsqueeze(0)
            
            # Positive similarity
            pos_sim = torch.mm(features[i].unsqueeze(0), positive_prototype.T) / self.temperature
            
            # Negative similarities (all other prototypes)
            neg_mask = torch.ones(self.num_classes, dtype=torch.bool)
            neg_mask[label] = False
            neg_prototypes = prototypes[neg_mask]
            neg_sim = torch.mm(features[i].unsqueeze(0), neg_prototypes.T) / self.temperature
            
            # Contrastive loss (InfoNCE)
            loss_i = -torch.log(
                torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum())
            )
            loss += loss_i
            
        return loss / batch_size
    
    def prototype_regularization_loss(self, features, labels):
        """
        Cross-Modal Prototypes Regularization (CMPR) Loss
        Eq. 5 from MFCPL: Shortens distance between local and global prototypes
        """
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.global_prototypes, dim=1)
        
        loss = 0.0
        for i in range(len(labels)):
            label = labels[i]
            loss += torch.norm(features[i] - prototypes[label], p=2).pow(2)
            
        return loss / len(labels)


# ====================== ENHANCED IMU CLIENT ======================
class EnhancedIMUClient(fl.client.NumPyClient):
    """
    Enhanced Client incorporating techniques from multiple papers:
    - Personalization (Haseeb et al., 2026)
    - Prototype Regularization (MFCPL, 2024)
    - Class Imbalance Handling (Albogamy, 2025)
    - Energy Awareness (EnFed, 2024)
    """
    def __init__(
        self, 
        train_subset, 
        val_subset, 
        client_id,
        num_classes,
        embedding_dim,
        config,
        battery_level=1.0,
        desired_accuracy=0.95
    ):
        self.client_id = client_id
        self.battery_level = battery_level
        self.desired_accuracy = desired_accuracy
        self.current_accuracy = 0.0
        self.personalization_epochs = 5
        
        # Model
        self.model = IMUTransformerEncoder(config).to(DEVICE)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=config["batch_size"],
            shuffle=True, 
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=config["batch_size"],
            shuffle=False, 
            num_workers=0
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        
        # Criterion with class weights for imbalance (Albogamy, 2025)
        self.criterion = self._get_weighted_criterion(train_subset)
        
        # Prototype Regularization (MFCPL, 2024)
        self.prototype_regularizer = PrototypeRegularization(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            temperature=0.1
        )
        
        # Personalization learning rate (Haseeb et al., 2026)
        self.personalization_lr = 1e-4
        
    def _get_weighted_criterion(self, dataset):
        """Weighted Cross-Entropy for class imbalance (Albogamy, 2025)"""
        # Compute class frequencies
        labels = [dataset[i]["label"] for i in range(len(dataset))]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        return torch.nn.NLLLoss(weight=class_weights.to(DEVICE))
    
    def get_parameters(self, config=None):
        """Get model parameters as NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays"""
        if hasattr(parameters, "tensors"):
            params_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        else:
            params_ndarrays = parameters
            
        state_dict = {
            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params_ndarrays)
        }
        self.model.load_state_dict(state_dict, strict=True)
    
    def _energy_aware_check(self):
        """Check if training should continue (EnFed, 2024)"""
        if self.battery_level < 0.2:  # 20% threshold
            print(f"Client {self.client_id}: Battery low ({self.battery_level:.2f}), stopping")
            return False
        if self.current_accuracy >= self.desired_accuracy:
            print(f"Client {self.client_id}: Desired accuracy reached ({self.current_accuracy:.4f})")
            return False
        return True
    
    def _extract_features_and_output(self, imu_data):
        """Extract both features and output from model"""
        # Assuming model returns (features, output) or just output
        result = self.model({"imu": imu_data})
        
        # If model returns tuple (features, logits)
        if isinstance(result, tuple):
            return result
        # If model returns only logits, we need to extract features separately
        else:
            # Get features from penultimate layer
            features = self.model.get_features({"imu": imu_data})
            return features, result
    
    def fit(self, parameters, config):
        """
        Local training with:
        - Standard FL training
        - Prototype regularization (MFCPL)
        - Personalization (Haseeb et al.)
        - Energy awareness (EnFed)
        """
        # Set global model parameters
        self.set_parameters(parameters)
        self.model.train()
        
        # Collect features for prototype update
        all_features = []
        all_labels = []
        total_loss = 0.0
        
        # ========== STAGE 1: Standard FL Training ==========
        for epoch in range(config.get("local_epochs", 5)):
            # Energy-aware check (EnFed, 2024)
            if not self._energy_aware_check():
                break
                
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()
                
                # Forward pass with feature extraction
                features, output = self._extract_features_and_output(imu)
                
                # 1. Cross-Entropy Loss
                ce_loss = self.criterion(output, labels)
                
                # 2. Prototype Regularization Loss (MFCPL, 2024)
                # Eq. 5: CMPR - Cross-Modal Prototypes Regularization
                proto_loss = self.prototype_regularizer.prototype_regularization_loss(
                    features, labels
                )
                
                # 3. Prototype Contrastive Loss (MFCPL, 2024)
                # Eq. 8: CMPC - Cross-Modal Prototypes Contrastive
                proto_contrastive_loss = self.prototype_regularizer.prototype_contrastive_loss(
                    features, labels
                )
                
                # Combined loss (Eq. 10 from MFCPL)
                alpha_reg = 0.1   # CMPR weight
                alpha_con = 0.2   # CMPC weight
                total_loss_batch = ce_loss + alpha_reg * proto_loss + alpha_con * proto_contrastive_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += ce_loss.item()
                
                # Store features for prototype update
                all_features.append(features.detach())
                all_labels.append(labels)
        
        # Update global prototypes with client features (MFCPL)
        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            self.prototype_regularizer.update_global_prototypes(all_features, all_labels)
        
        # ========== STAGE 2: Personalization (Haseeb et al., 2026) ==========
        self.personalize()
        
        # ========== STAGE 3: Evaluate current accuracy ==========
        self.current_accuracy = self._evaluate()
        
        return self.get_parameters(), len(self.train_loader.dataset), {
            "train_loss": total_loss / len(self.train_loader),
            "client_id": self.client_id,
            "current_accuracy": self.current_accuracy,
            "battery_level": self.battery_level
        }
    
    def personalize(self):
        """
        Per-client personalization through fine-tuning (Haseeb et al., 2026)
        Uses smaller learning rate for local adaptation
        """
        print(f"Client {self.client_id}: Starting personalization...")
        self.model.train()
        
        # Save original learning rate
        original_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = self.personalization_lr
        
        for epoch in range(self.personalization_epochs):
            for batch in self.train_loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()
                
                # Forward pass
                _, output = self._extract_features_and_output(imu)
                
                # Only CE loss for personalization
                loss = self.criterion(output, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Restore original learning rate
        self.optimizer.param_groups[0]['lr'] = original_lr
        print(f"Client {self.client_id}: Personalization complete")
    
    def _evaluate(self):
        """Evaluate current model on validation set"""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()
                
                _, output = self._extract_features_and_output(imu)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        return accuracy
    
    def evaluate(self, parameters, config):
        """Evaluate on test set"""
        self.set_parameters(parameters)
        self.model.eval()
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                imu = batch["imu"].to(DEVICE).float()
                labels = batch["label"].to(DEVICE).long()
                
                _, output = self._extract_features_and_output(imu)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        return accuracy, len(self.val_loader.dataset), {}


# ====================== GLOBAL PROTOTYPE AGGREGATION (MFCPL) ======================
class PrototypeAwareStrategy(fl.server.strategy.FedAvg):
    """
    Server strategy with prototype aggregation (MFCPL, 2024)
    """
    def __init__(self, num_classes, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.global_prototypes = torch.zeros(num_classes, embedding_dim)
        self.prototype_counts = torch.zeros(num_classes)
        
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client models and prototypes"""
        # Standard FedAvg aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is None:
            return aggregated
        
        # Aggregate prototypes from clients (MFCPL, 2024)
        # Eq. 4: Complete prototypes by averaging local prototypes
        client_prototypes = []
        for _, (_, client_metrics) in results:
            if 'prototypes' in client_metrics:
                client_prototypes.append(client_metrics['prototypes'])
        
        if client_prototypes:
            # Average prototypes across clients
            avg_prototypes = torch.stack(client_prototypes).mean(dim=0)
            self.global_prototypes = avg_prototypes
        
        return aggregated
    
    def get_global_prototypes(self):
        """Return global prototypes for distribution to clients"""
        return self.global_prototypes


# ====================== MAIN FUNCTION ======================
def main(train_csv: str, test_csv: str):
    # Load data
    _, test_dataset = load_data(train_csv, test_csv)
    client_datasets = split_train_data(train_csv, NUM_CLIENTS)
    
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Get model dimensions
    sample_model = IMUTransformerEncoder(config)
    embedding_dim = sample_model.get_embedding_dim()
    num_classes = config.get("num_classes", 6)
    
    def client_fn(context):
        if hasattr(context, "node_id"):
            cid = int(context.node_id)
        else:
            cid = int(context)
        
        # Safety check
        if cid >= len(client_datasets):
            cid = cid % len(client_datasets)
        
        print(f"Initializing Client {cid}")
        
        # Create train/val split for client
        train_size = int(0.8 * len(client_datasets[cid]))
        val_size = len(client_datasets[cid]) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            client_datasets[cid], [train_size, val_size]
        )
        
        return EnhancedIMUClient(
            train_subset=train_subset,
            val_subset=val_subset,
            client_id=cid,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            config=config,
            battery_level=1.0,  # Start with full battery
            desired_accuracy=0.95
        ).to_client()
    
    # Initialize strategy with prototype aggregation
    strategy = PrototypeAwareStrategy(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        test_loader=test_loader
    )
    
    print(f"Starting FL | {NUM_CLIENTS} Clients | {NUM_ROUNDS} Rounds\n")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.1 if torch.cuda.is_available() else 0},
    )


if __name__ == "__main__":
    main("train.csv", "test.csv")