import torch
import polars as pl
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import torch
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import polars as pl


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate predictions for a given dataloader.
    
    Parameters:
        model: The trained PyTorch model
        dataloader: DataLoader containing the data to predict on
        device: Device to run predictions on
        
    Returns:
        tuple: (y_actual, y_pred) as torch tensors
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            outputs = model(xb)
            
            # Handle output dimensions
            if outputs.ndim > yb.ndim:
                outputs = outputs.squeeze()
            
            all_preds.append(outputs.cpu())
            all_actuals.append(yb.cpu())
    
    y_pred = torch.cat(all_preds, dim=0)
    y_actual = torch.cat(all_actuals, dim=0)
    
    return y_actual, y_pred


def eval_regressor_performance(
    y_actual: torch.Tensor,
    y_pred: torch.Tensor,
    target_name: str = "target",
    threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for regression models.
    
    Parameters:
        y_actual: Actual target values (torch.Tensor)
        y_pred: Predicted target values (torch.Tensor)
        target_name: Name of the target variable
        threshold: Threshold for directional accuracy (default: 0.0 for mean)
        
    Returns:
        Dict containing performance metrics
    """
    # Convert to numpy
    y_actual_np = y_actual.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()
    
    # Basic regression metrics
    mse = mean_squared_error(y_actual_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual_np, y_pred_np)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_actual_np - y_pred_np))
    
    # Directional metrics
    actual_direction = (y_actual_np > threshold).astype(int)
    predicted_direction = (y_pred_np > threshold).astype(int)
    
    direction_accuracy = accuracy_score(actual_direction, predicted_direction)
    
    
    # Additional directional metrics
    # True positives: predicted up, actual up
    tp = np.sum((predicted_direction == 1) & (actual_direction == 1))
    # True negatives: predicted down, actual down
    tn = np.sum((predicted_direction == 0) & (actual_direction == 0))
    # False positives: predicted up, actual down
    fp = np.sum((predicted_direction == 1) & (actual_direction == 0))
    # False negatives: predicted down, actual up
    fn = np.sum((predicted_direction == 0) & (actual_direction == 1))
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Correlation
    correlation = np.corrcoef(y_actual_np, y_pred_np)[0, 1]
    
    # Explained variance (alternative to R²)
    explained_var = 1 - (np.var(y_actual_np - y_pred_np) / np.var(y_actual_np))
    
    metrics = {
        'target': target_name,
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'explained_variance': float(explained_var),
        'correlation': float(correlation),
        'direction_accuracy': float(direction_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'n_samples': len(y_actual_np)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Pretty print evaluation metrics.
    
    Parameters:
        metrics: Dictionary of metrics from eval_regressor_performance
    """
    print(f"\n{'='*60}")
    print(f"Model Performance Metrics - Target: {metrics['target']}")
    print(f"{'='*60}")
    
    print(f"\n📊 Regression Metrics:")
    print(f"  MSE:                {metrics['mse']:.6f}")
    print(f"  RMSE:               {metrics['rmse']:.6f}")
    print(f"  MAE:                {metrics['mae']:.6f}")
    print(f"  R²:                 {metrics['r2']:.6f}")
    print(f"  Explained Variance: {metrics['explained_variance']:.6f}")
    print(f"  Correlation:        {metrics['correlation']:.6f}")
    
    print(f"\n🎯 Directional Metrics:")
    print(f"  Accuracy:           {metrics['direction_accuracy']:.4f} ({metrics['direction_accuracy']*100:.2f}%)")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1_score']:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Positives:     {metrics['true_positives']}")
    print(f"  True Negatives:     {metrics['true_negatives']}")
    print(f"  False Positives:    {metrics['false_positives']}")
    print(f"  False Negatives:    {metrics['false_negatives']}")
    
    print(f"\n  Total Samples:      {metrics['n_samples']}")
    print(f"{'='*60}\n")


# Example usage:
if __name__ == "__main__":
    # Example of how to use these functions
    
    # In your evalPipeline or separate evaluation script:
    
    # Load trained model
    model = linearRegression(...)
    model.load_state_dict(torch.load('weights/model.pth'))
    
    # Generate predictions
    y_actual, y_pred = predict(model, testLoader, device='cuda')
    
    # Evaluate
    metrics = eval_regressor_performance(
        y_actual=y_actual,
        y_pred=y_pred,
        target_name='logRet',
        threshold=0.0  # Use 0 for mean-centered returns
    )
    
    # Print results
    print_metrics(metrics)
    
    # Or access specific metrics
   # print(f"Test MSE: {metrics['mse']:.6f}")
   # print(f"Direction Accuracy: {metrics['direction_accuracy']:.2%}")

    pass