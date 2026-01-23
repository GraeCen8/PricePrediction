#!/usr/bin/env python3
"""
Simple test to validate the new prediction mode configuration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests import load_config, update_configs_from_yaml

def test_config_loading():
    """Test loading the updated configuration."""
    print("=" * 60)
    print("TESTING CONFIGURATION LOADING")
    print("=" * 60)
    
    try:
        # Load the config
        config = load_config("config.yaml")
        update_configs_from_yaml(config)
        
        # Import DATA_CONFIG to check values
        from tests import DATA_CONFIG
        
        print("✅ Configuration loaded successfully!")
        print(f"Prediction mode: {DATA_CONFIG.get('prediction_mode', 'NOT SET')}")
        print(f"Future bar offset: {DATA_CONFIG.get('future_bar_offset', 'NOT SET')}")
        print(f"Window size: {DATA_CONFIG.get('window_size', 'NOT SET')}")
        
        # Validate prediction mode
        prediction_mode = DATA_CONFIG.get('prediction_mode')
        if prediction_mode in ['past', 'future']:
            print(f"✅ Valid prediction mode: {prediction_mode}")
        else:
            print(f"❌ Invalid prediction mode: {prediction_mode}")
        
        # Validate future bar offset
        offset = DATA_CONFIG.get('future_bar_offset')
        if isinstance(offset, int) and offset >= 0:
            print(f"✅ Valid future bar offset: {offset}")
        else:
            print(f"❌ Invalid future bar offset: {offset}")
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_config_loading()
