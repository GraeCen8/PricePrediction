import torch 
import torch.nn as nn

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Weight between MSE and direction
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # MSE component
        mse_loss = self.mse(y_pred, y_true)
        
        # Directional component (penalize wrong sign)
        direction_loss = torch.mean((torch.sign(y_pred) != torch.sign(y_true)).float())
        
        return self.alpha * mse_loss + (1 - self.alpha) * direction_loss
