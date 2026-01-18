import torch
import os
import glob
import re
from typing import Optional, Union


def load_pretrained_weights(model, weights_path: Optional[str] = None, 
                          weights_dir: str = "weights", 
                          model_type: str = None,
                          device: str = "cpu") -> bool:
    """
    Load pre-trained weights for a model from the weights directory.
    
    Args:
        model: PyTorch model to load weights into
        weights_path: Specific path to weights file. If None, will auto-detect.
        weights_dir: Directory containing weight files (default: "weights")
        model_type: Type of model (e.g., "SPHnet", "linearRegression") for auto-detection
        device: Device to load weights on
        
    Returns:
        bool: True if weights were loaded successfully, False otherwise
    """
    try:
        # Determine the weights file path
        if weights_path is not None:
            # Use specific path provided
            if not os.path.exists(weights_path):
                print(f"Warning: Weights file not found at {weights_path}")
                return False
            target_path = weights_path
        else:
            # Auto-detect weights file
            target_path = _auto_detect_weights(weights_dir, model_type)
            if target_path is None:
                print("Warning: No suitable weights file found for auto-detection")
                return False
        
        # Load weights
        print(f"Loading pre-trained weights from: {target_path}")
        
        # Load checkpoint to CPU first to avoid device issues
        checkpoint = torch.load(target_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check for common keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            # Assume it's directly a state dict
            state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict, strict=False)
        
        # Move model to target device
        model = model.to(device)
        
        print(f"✓ Successfully loaded pre-trained weights")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
        
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        return False


def _auto_detect_weights(weights_dir: str, model_type: str = None) -> Optional[str]:
    """
    Auto-detect the best weights file for a given model type.
    
    Args:
        weights_dir: Directory containing weight files
        model_type: Type of model to find weights for
        
    Returns:
        str: Path to detected weights file, or None if none found
    """
    if not os.path.exists(weights_dir):
        return None
    
    # Get all .pth files in the weights directory
    weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
    
    if not weight_files:
        return None
    
    if model_type is None:
        # If no model type specified, return the most recently modified file
        return max(weight_files, key=os.path.getmtime)
    
    # Try to find weights file matching the model type
    model_type_lower = model_type.lower()
    
    # First, try exact matches
    for weight_file in weight_files:
        filename = os.path.basename(weight_file).lower()
        if model_type_lower in filename:
            return weight_file
    
    # If no exact match, try to find the best candidate
    # Prefer files with "final", "best", or lowest loss
    candidates = []
    for weight_file in weight_files:
        filename = os.path.basename(weight_file).lower()
        score = 0
        
        # Check for quality indicators
        if "final" in filename:
            score += 10
        if "best" in filename:
            score += 8
        if "val" in filename:
            score += 5
        
        # Extract loss value if present (lower is better)
        loss_match = re.search(r'loss([\d.]+)', filename)
        if loss_match:
            loss_val = float(loss_match.group(1))
            score -= loss_val  # Lower loss = higher score
        
        candidates.append((score, weight_file))
    
    if candidates:
        # Return the file with highest score
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    # Fallback: return most recently modified
    return max(weight_files, key=os.path.getmtime)


def list_available_weights(weights_dir: str = "weights") -> list:
    """
    List all available weight files in the weights directory.
    
    Args:
        weights_dir: Directory containing weight files
        
    Returns:
        list: List of available weight files with metadata
    """
    if not os.path.exists(weights_dir):
        return []
    
    weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
    
    available = []
    for weight_file in weight_files:
        stat = os.stat(weight_file)
        available.append({
            'path': weight_file,
            'filename': os.path.basename(weight_file),
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified': stat.st_mtime
        })
    
    # Sort by modification time (most recent first)
    available.sort(key=lambda x: x['modified'], reverse=True)
    
    return available


def print_available_weights(weights_dir: str = "weights"):
    """Print available weight files in a formatted way."""
    available = list_available_weights(weights_dir)
    
    if not available:
        print(f"No weight files found in '{weights_dir}' directory")
        return
    
    print(f"\n📁 Available weight files in '{weights_dir}':")
    print("=" * 80)
    
    for i, weight_info in enumerate(available, 1):
        print(f"{i}. {weight_info['filename']}")
        print(f"   Size: {weight_info['size_mb']} MB")
        print(f"   Path: {weight_info['path']}")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    # Test the utility functions
    print("Testing weight loading utilities...")
    
    # List available weights
    print_available_weights()
    
    # Example usage (uncomment to test with actual model)
    # from models.SPHnet import SPHNet
    # model = SPHNet()
    # success = load_pretrained_weights(model, model_type="SPHnet")
    # print(f"Loading successful: {success}")