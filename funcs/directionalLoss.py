import torch 
import torch.nn as nn

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5, midpoint=0.0):
        super().__init__()
        self.alpha = alpha  # Weight between MSE and direction
        self.midpoint = midpoint  # Midpoint threshold for directional comparison
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # MSE component
        mse_loss = self.mse(y_pred, y_true)
        
        # Directional component (penalize wrong side of midpoint)
        pred_direction = torch.sign(y_pred - self.midpoint)
        true_direction = torch.sign(y_true - self.midpoint)
        direction_loss = torch.mean((pred_direction != true_direction).float())
        
        return self.alpha * mse_loss + (1 - self.alpha) * direction_loss