*this is to show how to use the eval pipeline *
# Key Improvements:
All hyperparameters at the top in clearly organized sections

Configuration templates for different experiment types

Helper functions for dynamic initialization of optimizers, loss functions, and schedulers

Better logging and experiment tracking

Flexible evaluation on multiple datasets (train/val/test)

Parameter counting and model summary

Prediction saving for further analysis

How to Use:
Quick test (10 epochs, small batch):

python
set_config_for_experiment("quick_test")
results = evalPipeline()
Full experiment (100 epochs, all features):

python
set_config_for_experiment("full_experiment")
results = evalPipeline()
Custom configuration:

python
# Override specific parameters
DATA_CONFIG["window_size"] = 96
TRAINING_CONFIG["learning_rate"] = 0.0005
MODEL_CONFIG["paramScale"] = 3

# Run with custom settings
results = evalPipeline()
Direct run (uses defaults at top):

python
# Just run with default configuration
if __name__ == '__main__':
    evalPipeline()
This structure makes it easy to:

Run different experiments by changing one line

Reproduce experiments exactly

Track what hyperparameters were used

Save predictions for further analysis

Switch between different model configurations