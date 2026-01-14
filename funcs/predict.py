import torch
import polars as pl
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def predict_regressor(model: torch.nn.Module, input_data: torch.Tensor) -> torch.Tensor:
    """
    Make a prediction using the trained model.
    
    Parameters:
        model (torch.nn.Module): Trained model.
        input_data (torch dataloader): Input tensor of shape [batchsize , 1, window_size, num_features].
        
    Returns:
        torch.Tensor: Prediction from the model.
    """
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_x, batch_y in input_data:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x)
            predictions.append(batch_pred)
    return torch.cat(predictions, dim=0).squeeze()