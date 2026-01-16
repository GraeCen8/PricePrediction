import torch
import torch.nn as nn 
from funcs.process import processing, LeanFeatureEngineering, TimeSeriesDataset, FullFeatureEngineering
from funcs.train import Training
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from models.linearRegression import linearRegression
import funcs.evalLoss as eval
import funcs.directionalLoss as d
import pandas as pd
from funcs.empty_scalar import EmptyScaler
import numpy as np

# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================

# --------------------
# DATA CONFIGURATION
# --------------------
DATA_CONFIG = {
    # Data source
    "from_file": True,
    "file_loc": "data",
    "timeframe": "15m",
    "pair": "btc",
    
    # Feature engineering
    "feature_mode": "lean",  # "lean" (20 features) or "full" (65+ features)
    "ema_n": 25,
    "fast_ema": 8,
    "slow_ema": 21,
    "vol_ema": 14,
    "mom_period": 14,
    
    # Preprocessing
    "target_column": "logRet",
    "window_size": 128,
    "split_ratio": "0.8,0.1,0.1",
    "normalizeFunc": MinMaxScaler,  # or StandardScaler, RobustScaler, MinMaxScaler
    "scale_target": False,
    "shuffle_train": False,
    
    # Dataloader
    "batch_size": 64,
    "num_workers": 2,
    "isClassifier": False,
}

# --------------------
# MODEL CONFIGURATION
# --------------------
MODEL_CONFIG = {
    "model_type": "linearRegression",  # Currently only linearRegression
    "inFeatures": None,  # Will be set dynamically from data
    "outFeatures": 1,
    "seq_len": DATA_CONFIG['window_size'],
    "paramScale": 5,
    
    # Activation functions (if applicable for other models)
    "activation": "gelu",  # "relu", "leaky_relu", "tanh", "sigmoid"
    "dropout_rate": 0.0,
    "batch_norm": False,
}

# --------------------
# TRAINING CONFIGURATION
# --------------------
TRAINING_CONFIG = {
    "optimizer": "adamw",  # "adam", "sgd", "rmsprop", "adamw"
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "momentum": 0.9,  # For SGD
    
    "criterion": "mse",  # "mse", "mae", "huber", "directional"
    "alpha": 0.5,  # For directional loss (mse vs direction weight)
    "midpoint": 0.0,  # For directional loss
    
    "scheduler": "cosine",  # "cosine", "step", "plateau", "none"
    "scheduler_params": {
        "T_max": 10,  # For cosine annealing
        "step_size": 5,  # For step LR
        "gamma": 0.1,  # For step LR
        "patience": 5,  # For ReduceLROnPlateau
        "factor": 0.5,  # For ReduceLROnPlateau
    },
    
    "epochs": 40,
    "valGap": 1,  # Validate every N epochs
    "saveGap": 0.1,  # Save checkpoint every N epochs
    "early_stopping_patience": 6,  # Stop if no improvement for N epochs
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "modelType": "regressor",  # "regressor" or "classifier"
    # TensorBoard configuration
    "use_tensorboard": True,
    "log_dir": "runs",
    "log_gradients": False,  # Can be heavy on memory
    "log_weights": True,
    "log_histograms": False,  # Can be heavy on memory
    
    # Early stopping
    "early_stopping_patience": 10,  # Stop if no improvement for N epochs
}

# --------------------
# EVALUATION CONFIGURATION
# --------------------
EVAL_CONFIG = {
    "threshold": 0.0,  # For directional accuracy (0 for mean-centered returns)
    "evaluate_on": ["val", "test"],  # Which datasets to evaluate on
    "metrics_to_print": ["mse", "rmse", "mae", "r2", "direction_accuracy"],
    "save_predictions": True,
    "plot_results": False,
}

# --------------------
# EXPERIMENT TRACKING
# --------------------
EXPERIMENT_CONFIG = {
    "experiment_name": "linear_regression_v1",
    "log_dir": "logs",
    "save_dir": "weights",
    "save_best": True,
    "save_last": True,
    "use_tensorboard": True,
    "use_csv_log": False,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_optimizer(model, config):
    """Initialize optimizer based on configuration."""
    params = model.parameters()
    
    if config["optimizer"] == "adam":
        return torch.optim.Adam(
            params, 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(
            params,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

def get_criterion(config):
    """Initialize loss function based on configuration."""
    if config["criterion"] == "mse":
        return nn.MSELoss()
    elif config["criterion"] == "mae":
        return nn.L1Loss()
    elif config["criterion"] == "huber":
        return nn.HuberLoss()
    elif config["criterion"] == "directional":
        return d.DirectionalLoss(alpha=config["alpha"], midpoint=config["midpoint"])
    else:
        raise ValueError(f"Unknown criterion: {config['criterion']}")

def get_scheduler(optimizer, config):
    """Initialize learning rate scheduler based on configuration."""
    if config["scheduler"] == "none" or config["scheduler"] is None:
        return None
    elif config["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["scheduler_params"]["T_max"],
            last_epoch=-1
        )
    elif config["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler_params"]["step_size"],
            gamma=config["scheduler_params"]["gamma"]
        )
    elif config["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config["scheduler_params"]["patience"],
            factor=config["scheduler_params"]["factor"],
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")

def get_feature_engineering_func(config):
    """Get feature engineering function based on configuration."""
    def custom_feature_engineering(df):
        # You can customize this based on your feature engineering functions
        if config["feature_mode"] == "lean":
            # Import here to avoid circular imports
            from funcs.process import LeanFeatureEngineering
            return LeanFeatureEngineering(
                df, 
                fast_ema=config["fast_ema"],
                slow_ema=config["slow_ema"],
                vol_ema=config["vol_ema"],
                mom_period=config["mom_period"]
            )
        elif config["feature_mode"] == "full":
            from funcs.process import FullFeatureEngineering
            return FullFeatureEngineering(
                df,
                fast_ema=config["fast_ema"],
                slow_ema=config["slow_ema"],
                vol_ema=config["vol_ema"],
                mom_period=config["mom_period"]
            )
        else:
            # Use the example function if provided in process.py
            return FeatureEngineeringExample(df, ema_n=config["ema_n"])
    return custom_feature_engineering

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evalPipeline():
    """
    Main evaluation pipeline with all hyperparameters configurable at the top.
    """
    print("=" * 60)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 60)
    
    # Print configuration summary
    print("\n📊 CONFIGURATION SUMMARY:")
    print(f"  Data: {DATA_CONFIG['pair']} {DATA_CONFIG['timeframe']}")
    print(f"  Features: {DATA_CONFIG['feature_mode']} mode")
    print(f"  Model: {MODEL_CONFIG['model_type']}")
    print(f"  Training: {TRAINING_CONFIG['epochs']} epochs")
    print(f"  Device: {TRAINING_CONFIG['device']}")
    
    # ==================== DATA PROCESSING ====================
    print("\n" + "=" * 60)
    print("STEP 1: DATA PROCESSING")
    print("=" * 60)
    
    data_processing_params = {
        "from_file": DATA_CONFIG["from_file"],
        "file_loc": DATA_CONFIG["file_loc"],
        "timeframe": DATA_CONFIG["timeframe"],
        "pair": DATA_CONFIG["pair"],
        "dataframe": None,
        "date_column": "timestamp",
        "dataProcessFunc": get_feature_engineering_func(DATA_CONFIG),
        "feature_mode": DATA_CONFIG["feature_mode"],
        "normalizeFunc": DATA_CONFIG["normalizeFunc"],
        "isClassifier": DATA_CONFIG["isClassifier"],
        "datasetClass": TimeSeriesDataset,
        "batch_size": DATA_CONFIG["batch_size"],
        "num_workers": DATA_CONFIG["num_workers"],
        "target_column": DATA_CONFIG["target_column"],
        "window_size": DATA_CONFIG["window_size"],
        "split_ratio": DATA_CONFIG["split_ratio"],
        "shuffle_train": DATA_CONFIG["shuffle_train"],
        "scale_target": DATA_CONFIG["scale_target"],
        "verbose": True,
    }
    
    processor = processing(**data_processing_params)
    trainLoader, valLoader, testLoader, feature_scaler, target_scaler, feature_names, metadata = processor.process()
    
    # Update model config with actual feature count
    MODEL_CONFIG["inFeatures"] = 12#metadata['n_features']
    print(f"\n  Number of features detected: {MODEL_CONFIG['inFeatures']}")
    print(f"  Feature names: {feature_names[:5]}... (first 5 of {len(feature_names)})")
    
    # ==================== MODEL INITIALIZATION ====================
    print("\n" + "=" * 60)
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 60)
    
    if MODEL_CONFIG["model_type"] == "linearRegression":
        model = linearRegression(
            inFeatures=MODEL_CONFIG["inFeatures"],
            outFeatures=MODEL_CONFIG["outFeatures"],
            seq_len=MODEL_CONFIG["seq_len"],
            paramScale=MODEL_CONFIG["paramScale"]
        )
    else:
        raise ValueError(f"Unsupported model type: {MODEL_CONFIG['model_type']}")
    
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Input shape: (batch, {MODEL_CONFIG['seq_len']}, {MODEL_CONFIG['inFeatures']})")
    print(f"  Output shape: (batch, {MODEL_CONFIG['outFeatures']})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ==================== TRAINING SETUP ====================
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING SETUP")
    print("=" * 60)
    
    # Initialize optimizer, criterion, and scheduler
    optimizer = get_optimizer(model, TRAINING_CONFIG)
    criterion = get_criterion(TRAINING_CONFIG)
    scheduler = get_scheduler(optimizer, TRAINING_CONFIG)
    
    print(f"  Optimizer: {TRAINING_CONFIG['optimizer']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Loss function: {TRAINING_CONFIG['criterion']}")
    print(f"  Scheduler: {TRAINING_CONFIG['scheduler']}")
    
    # ==================== TRAINING ====================
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING")
    print("=" * 60)
    
    training_params = {
    "model": model,
    "optimizer": optimizer,
    "trainLoader": trainLoader,
    "valLoader": valLoader,
    "testLoader": testLoader,
    "scheduler": scheduler,
    "epochs": TRAINING_CONFIG["epochs"],
    "valGap": TRAINING_CONFIG["valGap"],
    "saveGap": TRAINING_CONFIG["saveGap"],
    "criterion": criterion,
    "device": TRAINING_CONFIG["device"],
    "modelType": TRAINING_CONFIG["modelType"],
    
    # New parameters for TensorBoard and tracking
    "experiment_name": EXPERIMENT_CONFIG["experiment_name"],
    "use_tensorboard": TRAINING_CONFIG["use_tensorboard"],
    "log_dir": TRAINING_CONFIG["log_dir"],
    "log_gradients": TRAINING_CONFIG["log_gradients"],
    "log_weights": TRAINING_CONFIG["log_weights"],
    "log_histograms": TRAINING_CONFIG["log_histograms"],
    "early_stopping_patience": TRAINING_CONFIG["early_stopping_patience"],
}
    
    trainer = Training(**training_params)
    trainer.train()
    
    # ==================== EVALUATION ====================
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATION")
    print("=" * 60)
    
    # Move model to evaluation device
    model = model.to(TRAINING_CONFIG["device"])
    
    # Evaluate on specified datasets
    for dataset_name in EVAL_CONFIG["evaluate_on"]:
        if dataset_name == "val":
            dataloader = valLoader
            dataset_desc = "Validation"
        elif dataset_name == "test":
            dataloader = testLoader
            dataset_desc = "Test"
        elif dataset_name == "train":
            dataloader = trainLoader
            dataset_desc = "Training"
        else:
            continue
        
        print(f"\n📈 {dataset_desc.upper()} SET EVALUATION:")
        y_actual, y_pred = eval.predict(model, dataloader, device=TRAINING_CONFIG["device"])
        
        metrics = eval.eval_regressor_performance(
            y_actual=y_actual,
            y_pred=y_pred,
            target_name=DATA_CONFIG["target_column"],
            threshold=EVAL_CONFIG["threshold"]
        )
        
        eval.print_metrics(metrics)
        
        # Save predictions if configured
        if EVAL_CONFIG["save_predictions"]:
            predictions_df = pd.DataFrame({
                'actual': y_actual.cpu().numpy().flatten(),
                'predicted': y_pred.cpu().numpy().flatten()
            })
            pred_file = f"{EXPERIMENT_CONFIG['log_dir']}/{EXPERIMENT_CONFIG['experiment_name']}_{dataset_name}_predictions.csv"
            predictions_df.to_csv(pred_file, index=False)
            print(f"  Predictions saved to: {pred_file}")
    
    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        "model": model,
        "metrics": metrics,
        "feature_names": feature_names,
        "metadata": metadata
    }


# ============================================================================
# CONFIGURATION TEMPLATES FOR DIFFERENT EXPERIMENTS
# ============================================================================

def set_config_for_experiment(experiment_name):
    """
    Pre-configured settings for different types of experiments.
    Call this BEFORE evalPipeline() to switch configurations.
    """
    global DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, EVAL_CONFIG, EXPERIMENT_CONFIG
    
    if experiment_name == "quick_test":
        """Quick test with minimal settings."""
        DATA_CONFIG.update({
            "window_size": 64,
            "batch_size": 32,
            "feature_mode": "lean",
        })
        TRAINING_CONFIG.update({
            "epochs": 10,
            "learning_rate": 0.001,
        })
        EXPERIMENT_CONFIG.update({
            "experiment_name": "quick_test",
        })
        
    elif experiment_name == "full_experiment":
        """Full experiment with all features."""
        DATA_CONFIG.update({
            "window_size": 128,
            "batch_size": 64,
            "feature_mode": "full",
            "normalizeFunc": RobustScaler,
            "scale_target": True,
        })
        TRAINING_CONFIG.update({
            "epochs": 100,
            "learning_rate": 0.0001,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "early_stopping_patience": 15,
        })
        MODEL_CONFIG.update({
            "paramScale": 8,
        })
        EXPERIMENT_CONFIG.update({
            "experiment_name": "full_experiment",
            "use_tensorboard": True,
        })
        
    elif experiment_name == "directional_focus":
        """Focus on directional accuracy."""
        DATA_CONFIG.update({
            "feature_mode": "lean",
            "normalizeFunc": StandardScaler,
        })
        TRAINING_CONFIG.update({
            "criterion": "directional",
            "alpha": 0.3,  # More weight on direction
            "learning_rate": 0.0005,
        })
        EVAL_CONFIG.update({
            "threshold": 0.0,
        })
        EXPERIMENT_CONFIG.update({
            "experiment_name": "directional_focus",
        })
        
    else:
        print(f"Unknown experiment template: {experiment_name}")
        print("Using default configuration.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    # Option 1: Use default configuration (as defined at top)
    # results = evalPipeline()
    
    # Option 2: Use a pre-configured experiment template
    set_config_for_experiment("quick_test")  # Uncomment to use quick test
    # set_config_for_experiment("full_experiment")  # Uncomment for full experiment
    # set_config_for_experiment("directional_focus")  # Uncomment for directional focus
    DATA_CONFIG["normalizeFunc"] = StandardScaler
    TRAINING_CONFIG["alpha"] = 0.2
    TRAINING_CONFIG["epochs"] = 3
    TRAINING_CONFIG["learning_rate"] = 0.0005
    TRAINING_CONFIG["scheduler"] = "none"
    TRAINING_CONFIG["early_stopping_patience"] = 3
    TRAINING_CONFIG["weight_decay"] = 0.0
    TRAINING_CONFIG["criterion"] = "directional"
    MODEL_CONFIG["paramScale"] = 5
    TRAINING_CONFIG['alpha'] = 0.3
    TRAINING_CONFIG['midpoint'] = 0.5
    

    # Run the pipeline
    results = evalPipeline()
    
    # Print final summary
    print("\n🎯 FINAL SUMMARY:")
    print(f"  Model saved in: {EXPERIMENT_CONFIG['save_dir']}/")
    print(f"  Logs saved in: {EXPERIMENT_CONFIG['log_dir']}/")
    print(f"  Features used: {len(results['feature_names'])}")
    print(f"  Sample size - Train: {results['metadata']['n_train_samples']:,}")
    print(f"  Sample size - Val: {results['metadata']['n_val_samples']:,}")
    print(f"  Sample size - Test: {results['metadata']['n_test_samples']:,}")