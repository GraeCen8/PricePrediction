#!/usr/bin/env python3
"""
Test script for past vs future bar prediction functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from funcs.process import processing
import pandas as pd
import numpy as np

def create_test_data(n_samples=1000):
    """Create synthetic OHLC data for testing."""
    np.random.seed(42)
    
    # Generate synthetic price data
    price = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))
    
    # Create OHLC data
    data = {
        'open': price + np.random.normal(0, 0.1, n_samples),
        'high': price + np.abs(np.random.normal(0.2, 0.1, n_samples)),
        'low': price - np.abs(np.random.normal(0.2, 0.1, n_samples)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def test_prediction_modes():
    """Test both past and future prediction modes."""
    print("=" * 60)
    print("TESTING PREDICTION MODES")
    print("=" * 60)
    
    # Create test data
    df = create_test_data(500)
    print(f"Created test data: {len(df)} rows")
    
    # Test parameters
    test_params = {
        'from_file': False,
        'dataframe': df,
        'window_size': 10,
        'batch_size': 32,
        'feature_mode': 'lean',
        'verbose': True
    }
    
    # Test 1: Past prediction (traditional)
    print("\n1. TESTING PAST PREDICTION MODE:")
    print("-" * 40)
    
    past_params = test_params.copy()
    past_params['prediction_mode'] = 'past'
    
    try:
        processor_past = processing(**past_params)
        train_loader, val_loader, test_loader, _, _, _, metadata = processor_past.process()
        
        # Get a sample batch to check shapes
        for X, y in train_loader:
            print(f"Past mode - Input shape: {X.shape}, Target shape: {y.shape}")
            print(f"Sample target values: {y[:5].numpy().flatten()}")
            break
            
    except Exception as e:
        print(f"Error in past mode: {e}")
    
    # Test 2: Future prediction with offset 1
    print("\n2. TESTING FUTURE PREDICTION MODE (offset=1):")
    print("-" * 40)
    
    future_params_1 = test_params.copy()
    future_params_1['prediction_mode'] = 'future'
    future_params_1['future_bar_offset'] = 1
    
    try:
        processor_future_1 = processing(**future_params_1)
        train_loader, val_loader, test_loader, _, _, _, metadata = processor_future_1.process()
        
        # Get a sample batch to check shapes
        for X, y in train_loader:
            print(f"Future mode (offset=1) - Input shape: {X.shape}, Target shape: {y.shape}")
            print(f"Sample target values: {y[:5].numpy().flatten()}")
            break
            
    except Exception as e:
        print(f"Error in future mode (offset=1): {e}")
    
    # Test 3: Future prediction with offset 5
    print("\n3. TESTING FUTURE PREDICTION MODE (offset=5):")
    print("-" * 40)
    
    future_params_5 = test_params.copy()
    future_params_5['prediction_mode'] = 'future'
    future_params_5['future_bar_offset'] = 5
    
    try:
        processor_future_5 = processing(**future_params_5)
        train_loader, val_loader, test_loader, _, _, _, metadata = processor_future_5.process()
        
        # Get a sample batch to check shapes
        for X, y in train_loader:
            print(f"Future mode (offset=5) - Input shape: {X.shape}, Target shape: {y.shape}")
            print(f"Sample target values: {y[:5].numpy().flatten()}")
            break
            
    except Exception as e:
        print(f"Error in future mode (offset=5): {e}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_prediction_modes()
