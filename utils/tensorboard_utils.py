"""
Script to monitor training progress in real-time.
Run this in a separate terminal while training.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

def monitor_training(experiment_name, update_interval=30):
    """
    Monitor training progress by reading the CSV log file.
    
    Args:
        experiment_name: Name of the experiment to monitor
        update_interval: Seconds between updates
    """
    
    log_dir = f"logs/linearRegression_{experiment_name}"
    csv_path = os.path.join(log_dir, "training_log.csv")
    
    print(f"Monitoring training for experiment: {experiment_name}")
    print(f"Log file: {csv_path}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            if os.path.exists(csv_path):
                # Read the CSV
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    clear_output(wait=True)
                    
                    # Create monitoring dashboard
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Loss plot
                    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
                    if 'val_loss' in df.columns and df['val_loss'].notna().any():
                        val_data = df[df['val_loss'].notna()]
                        axes[0].plot(val_data['epoch'], val_data['val_loss'], 
                                   label='Val Loss', marker='s')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].set_title('Training Progress')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Learning rate plot
                    if 'learning_rate' in df.columns:
                        axes[1].plot(df['epoch'], df['learning_rate'], color='green', marker='o')
                        axes[1].set_xlabel('Epoch')
                        axes[1].set_ylabel('Learning Rate')
                        axes[1].set_title('Learning Rate Schedule')
                        axes[1].set_yscale('log')
                        axes[1].grid(True, alpha=0.3)
                    
                    # R² score plot (if available)
                    if 'r2_score' in df.columns and df['r2_score'].notna().any():
                        r2_data = df[df['r2_score'].notna()]
                        axes[2].plot(r2_data['epoch'], r2_data['r2_score'], color='purple', marker='o')
                        axes[2].set_xlabel('Epoch')
                        axes[2].set_ylabel('R² Score')
                        axes[2].set_title('Validation R² Score')
                        axes[2].grid(True, alpha=0.3)
                        axes[2].set_ylim(-1, 1)
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Print latest metrics
                    latest = df.iloc[-1]
                    print(f"Latest Epoch: {int(latest['epoch'])}")
                    print(f"Train Loss: {latest['train_loss']:.6f}")
                    if 'val_loss' in latest and pd.notna(latest['val_loss']):
                        print(f"Val Loss: {latest['val_loss']:.6f}")
                    if 'r2_score' in latest and pd.notna(latest['r2_score']):
                        print(f"R² Score: {latest['r2_score']:.4f}")
                    print(f"Learning Rate: {latest.get('learning_rate', 'N/A')}")
                    print(f"\nMonitoring... Next update in {update_interval} seconds")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"Error in monitoring: {e}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--experiment', type=str, default='experiment',
                       help='Experiment name to monitor')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds')
    
    args = parser.parse_args()
    monitor_training(args.experiment, args.interval)