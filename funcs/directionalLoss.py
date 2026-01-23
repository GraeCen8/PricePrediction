import torch 
import torch.nn as nn
import json
import os

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.0, midpoint=0.0, temperature=1.0, normalize_mse=False, balance_reg=0.0, focal_gamma=0.0):
        """
        Directional loss that directly optimizes for balanced accuracy.
        Uses a combination of BCE with strong distribution matching.
        """
        super().__init__()
        self.alpha = alpha
        self.midpoint = midpoint
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        
        # Convert to binary classification
        true_binary = (y_true > self.midpoint).float()
        pred_binary = (y_pred > self.midpoint).float()
        
        # Calculate true class distribution
        true_ratio = torch.mean(true_binary)
        pred_ratio = torch.mean(pred_binary)
        
        # Calculate class weights based on true distribution
        pos_count = torch.sum(true_binary)
        neg_count = batch_size - pos_count
        
        if pos_count > 0 and neg_count > 0:
            # Inverse frequency weighting - very strong
            pos_weight = (neg_count / pos_count).item()
            neg_weight = (pos_count / neg_count).item()
            # Cap weights to prevent extreme values
            pos_weight = min(pos_weight, 10.0)
            neg_weight = min(neg_weight, 10.0)
        else:
            pos_weight = 1.0
            neg_weight = 1.0
        
        # Create weight tensor
        class_weights = true_binary * pos_weight + (1 - true_binary) * neg_weight
        
        # Binary cross-entropy with class weights
        pred_logits = y_pred - self.midpoint
        sigmoid_logits = torch.sigmoid(pred_logits)
        bce_per_sample = -(true_binary * torch.log(sigmoid_logits + 1e-8) + 
                          (1 - true_binary) * torch.log(1 - sigmoid_logits + 1e-8))
        bce_loss = torch.mean(bce_per_sample * class_weights)
        
        # CRITICAL: Distribution matching loss - this is the key
        # Force prediction distribution to match true distribution
        # Use KL divergence style loss
        ratio_diff = torch.abs(pred_ratio - true_ratio)
        # Make this a reasonable component of the loss (not overwhelming)
        distribution_loss = 10 * ratio_diff  # Much more reasonable
        
        # Also ensure predictions have diversity (not all the same)
        pred_std = torch.std(y_pred)
        true_std = torch.std(y_true) + 1e-8
        std_ratio = pred_std / true_std
        # Penalize if std is too low (all predictions similar)
        diversity_loss = 1.0 * torch.clamp(1.0 - std_ratio, min=0.0)
        
        # Combined loss - distribution matching is critical
        direction_loss = bce_loss + distribution_loss + diversity_loss
        
        # Optional MSE component
        if self.alpha > 0:
            mse_loss = self.mse(y_pred, y_true)
            mean_abs = torch.mean(torch.abs(y_true)) + 1e-8
            mse_normalized = mse_loss / (mean_abs ** 2 + 1e-8)
            total_loss = self.alpha * mse_normalized + (1 - self.alpha) * direction_loss
        else:
            total_loss = direction_loss
        
        # #region agent log
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 0:
            try:
                log_dir = '/home/grae/Coding/PricePrediction/.cursor'
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, 'debug.log')
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "directionalLoss.py:forward",
                        "message": "Loss components",
                        "data": {
                            "batch_num": self._log_counter,
                            "pred_ratio": float(pred_ratio.item()),
                            "true_ratio": float(true_ratio.item()),
                            "ratio_diff": float(ratio_diff.item()),
                            "distribution_loss": float(distribution_loss.item()),
                            "diversity_loss": float(diversity_loss.item()),
                            "bce_loss": float(bce_loss.item()),
                            "direction_loss": float(direction_loss.item()),
                            "total_loss": float(total_loss.item()),
                            "pred_std": float(pred_std.item()),
                            "true_std": float(true_std.item())
                        },
                        "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
                    }) + "\n")
            except Exception as e:
                pass
        # #endregion
        
        return total_loss
