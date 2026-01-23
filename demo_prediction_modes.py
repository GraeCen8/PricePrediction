#!/usr/bin/env python3
"""
Demonstration of the new past/future bar prediction functionality.
This script shows how to use the updated configuration to switch between
prediction modes without changing code.
"""

# Example usage configurations:

# 1. Traditional past bar prediction (next bar after window)
PAST_CONFIG = """
data:
  prediction_mode: "past"
  window_size: 64
  # ... other config
"""

# 2. Future bar prediction with 1 bar offset
FUTURE_1_CONFIG = """
data:
  prediction_mode: "future"
  future_bar_offset: 1
  window_size: 64
  # ... other config
"""

# 3. Future bar prediction with 5 bar offset
FUTURE_5_CONFIG = """
data:
  prediction_mode: "future"
  future_bar_offset: 5
  window_size: 64
  # ... other config
"""

def demonstrate_prediction_modes():
    """
    Demonstrate the different prediction modes.
    
    The key changes made to the codebase:
    
    1. Updated config.yaml with new parameters:
       - prediction_mode: "past" or "future"
       - future_bar_offset: number of bars to look ahead
    
    2. Modified funcs/process.py:
       - Added prediction_mode and future_bar_offset parameters
       - Updated slide() method to handle both modes
       - Added verbose logging for prediction settings
    
    3. Updated tests.py:
       - Added new parameters to data_processing_params
       - Ensured configuration is passed through correctly
    
    How it works:
    - Past mode: target = data[window_end] (traditional next bar)
    - Future mode: target = data[window_end + future_bar_offset]
    
    This allows you to experiment with different prediction horizons
    without changing the model architecture or training code.
    """
    
    print("=" * 60)
    print("PAST/FUTURE BAR PREDICTION DEMONSTRATION")
    print("=" * 60)
    
    print("\n📋 CONFIGURATION OPTIONS:")
    print("1. Set prediction_mode: 'past' for traditional next-bar prediction")
    print("2. Set prediction_mode: 'future' for future-bar prediction")
    print("3. Set future_bar_offset: number of bars to look ahead")
    
    print("\n🔧 HOW TO USE:")
    print("Edit config.yaml and set:")
    print("  data:")
    print("    prediction_mode: 'future'  # or 'past'")
    print("    future_bar_offset: 5       # look 5 bars ahead")
    
    print("\n💡 EXAMPLES:")
    print("- prediction_mode: 'past' → Predict next bar (traditional)")
    print("- prediction_mode: 'future', future_bar_offset: 1 → Predict 1 bar ahead")
    print("- prediction_mode: 'future', future_bar_offset: 10 → Predict 10 bars ahead")
    
    print("\n🎯 BENEFITS:")
    print("- Easy experimentation with different prediction horizons")
    print("- No code changes needed to switch modes")
    print("- Same model architecture works for all modes")
    print("- Configuration-driven approach")
    
    print("\n📊 TECHNICAL DETAILS:")
    print("- Window size determines input context")
    print("- Future offset determines prediction horizon")
    print("- Data windows are adjusted to prevent out-of-bounds errors")
    print("- Verbose logging shows prediction mode and offset")
    
    print("\n" + "=" * 60)
    print("READY TO TEST!")
    print("Run: python tests.py")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_prediction_modes()
