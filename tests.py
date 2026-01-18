import torch
import torch.nn as nn 
from funcs.process import processing, LeanFeatureEngineering, TimeSeriesDataset, FullFeatureEngineering
from funcs.train import Training
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from models.linearRegression import linearRegression
from models.LSTM import LSTMModel
import importlib
cnn_lstm_module = importlib.import_module('models.CNN-LSTM')
CNNLSTM = cnn_lstm_module.CNNLSTM
from models.TCN import TCNModel
from models.SPHnet import SPHNet
from models.TimeSeriesTransformer import TimeSeriesTransformer
import funcs.evalLoss as eval
import funcs.directionalLoss as d
import pandas as pd
from funcs.empty_scalar import EmptyScaler
import numpy as np
import yaml
from pathlib import Path

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path="config.yaml", template_name=None):
    """
    Load configuration from YAML file with optional template override.
    
    Args:
        config_path: Path to the YAML configuration file
        template_name: Optional template name to apply (e.g., 'quick_test', 'full_experiment')
    
    Returns:
        dict: Loaded configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply template if specified
    if template_name and 'templates' in config and template_name in config['templates']:
        template = config['templates'][template_name]
        for section, settings in template.items():
            if section in config:
                config[section].update(settings)
    
    return config

def update_configs_from_yaml(config):
    """
    Update global configuration dictionaries from loaded YAML config.
    
    Args:
        config: Loaded configuration dictionary
    """
    global DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, EVAL_CONFIG, EXPERIMENT_CONFIG
    
    # Update each configuration section
    if 'data' in config:
        DATA_CONFIG.update(config['data'])
    
    if 'model' in config:
        MODEL_CONFIG.update(config['model'])
    
    if 'training' in config:
        TRAINING_CONFIG.update(config['training'])
    
    if 'evaluation' in config:
        EVAL_CONFIG.update(config['evaluation'])
    
    if 'experiment' in config:
        EXPERIMENT_CONFIG.update(config['experiment'])

def get_scaler_from_string(scaler_name):
    """
    Convert scaler string name to actual scaler class.
    
    Args:
        scaler_name: String name of the scaler
    
    Returns:
        Scaler class
    """
    scaler_map = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "EmptyScaler": EmptyScaler
    }
    
    if scaler_name not in scaler_map:
        raise ValueError(f"Unknown scaler: {scaler_name}. Available: {list(scaler_map.keys())}")
    
    return scaler_map[scaler_name]

# ============================================================================
# HYPERPARAMETERS & CONFIGURATION (DEFAULTS)
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
    "model_type": "linearRegression",  # Options: "linearRegression", "LSTM", "CNN-LSTM", "TCN", "SPHnet", "TimeSeriesTransformer"
    "inFeatures": None,  # Will be set dynamically from data
    "outFeatures": 1,
    "seq_len": DATA_CONFIG['window_size'],
    
    # linearRegression parameters
    "paramScale": 5,
    
    # LSTM parameters
    "lstm_hidden_size": 128,
    "lstm_num_layers": 3,
    "lstm_dropout": 0.2,
    
    # CNN-LSTM parameters
    "cnn_channels": 64,
    "cnn_lstm_hidden": 128,
    "cnn_lstm_dropout": 0.2,
    
    # TCN parameters
    "tcn_num_channels": [64, 128, 128, 256],
    "tcn_kernel_size": 3,
    "tcn_dropout": 0.2,
    
    # SPHnet parameters
    "sphnet_patch_size": 8,
    "sphnet_embed_dim": 128,
    "sphnet_vit_num_layers": 4,
    "sphnet_transformer_num_layers": 4,
    "sphnet_num_heads": 8,
    "sphnet_ff_dim": 512,
    "sphnet_dropout": 0.1,
    
    # TimeSeriesTransformer parameters
    "transformer_d_model": 128,
    "transformer_nhead": 8,
    "transformer_num_layers": 4,
    "transformer_dim_feedforward": 512,
    "transformer_dropout": 0.1,
    
    # Common parameters
    "dropout_rate": 0.0,  # General dropout (used if model-specific not set)
    "activation": "gelu",  # "relu", "leaky_relu", "tanh", "sigmoid"
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
    "alpha": 0.3,  # For directional loss (mse vs direction weight, lower = more focus on direction)
    "midpoint": 0.0,  # For directional loss
    "temperature": 10.0,  # For directional loss smoothness (higher = sharper)
    "normalize_mse": True,  # Normalize MSE to match directional loss scale
    
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
    "saveGap": 10,  # Save checkpoint every N epochs (must be integer)
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
        return d.DirectionalLoss(
            alpha=config.get("alpha", 0.1),
            midpoint=config.get("midpoint", 0.0),
            temperature=config.get("temperature", 3.0),
            normalize_mse=config.get("normalize_mse", True),
            balance_reg=config.get("balance_reg", 10.0),
            focal_gamma=config.get("focal_gamma", 2.0)
        )
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
            raise ValueError(f"Unknown feature_mode: {config['feature_mode']}. "
                           f"Supported modes: 'lean', 'full'")
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
    
    # Update model config with actual feature count and window size
    MODEL_CONFIG["inFeatures"] = metadata['n_features']
    MODEL_CONFIG["seq_len"] = metadata['window_size']  # Ensure seq_len matches actual window_size
    
    print(f"\n  Number of features detected: {MODEL_CONFIG['inFeatures']}")
    print(f"  Feature names: {feature_names[:5]}... (first 5 of {len(feature_names)})")
    
    # ==================== MODEL INITIALIZATION ====================
    print("\n" + "=" * 60)
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 60)
    
    # Initialize model based on model_type
    model_type = MODEL_CONFIG["model_type"]
    n_features = MODEL_CONFIG["inFeatures"]
    
    if model_type == "linearRegression":
        model = linearRegression(
            inFeatures=n_features,
            outFeatures=MODEL_CONFIG["outFeatures"],
            seq_len=MODEL_CONFIG["seq_len"],
            paramScale=MODEL_CONFIG["paramScale"]
        )
    elif model_type == "LSTM":
        model = LSTMModel(
            n_features=n_features,
            hidden_size=MODEL_CONFIG["lstm_hidden_size"],
            num_layers=MODEL_CONFIG["lstm_num_layers"],
            dropout=MODEL_CONFIG["lstm_dropout"]
        )
    elif model_type == "CNN-LSTM":
        model = CNNLSTM(
            n_features=n_features,
            cnn_channels=MODEL_CONFIG["cnn_channels"],
            lstm_hidden=MODEL_CONFIG["cnn_lstm_hidden"],
            dropout=MODEL_CONFIG["cnn_lstm_dropout"]
        )
    elif model_type == "TCN":
        model = TCNModel(
            n_features=n_features,
            num_channels=MODEL_CONFIG["tcn_num_channels"],
            kernel_size=MODEL_CONFIG["tcn_kernel_size"],
            dropout=MODEL_CONFIG["tcn_dropout"]
        )
    elif model_type == "SPHnet":
        # Ensure seq_len is divisible by patch_size
        patch_size = MODEL_CONFIG["sphnet_patch_size"]
        seq_len = MODEL_CONFIG["seq_len"]
        if seq_len % patch_size != 0:
            # Adjust seq_len to be divisible by patch_size
            seq_len = (seq_len // patch_size) * patch_size
            MODEL_CONFIG["seq_len"] = seq_len
            print(f"  Warning: Adjusted seq_len to {seq_len} to be divisible by patch_size {patch_size}")
        
        model = SPHNet(
            num_features=n_features,
            patch_size=MODEL_CONFIG["sphnet_patch_size"],
            embed_dim=MODEL_CONFIG["sphnet_embed_dim"],
            vit_num_layers=MODEL_CONFIG["sphnet_vit_num_layers"],
            transformer_num_layers=MODEL_CONFIG["sphnet_transformer_num_layers"],
            num_heads=MODEL_CONFIG["sphnet_num_heads"],
            ff_dim=MODEL_CONFIG["sphnet_ff_dim"],
            dropout=MODEL_CONFIG["sphnet_dropout"],
            output_dim=MODEL_CONFIG["outFeatures"]
        )
    elif model_type == "TimeSeriesTransformer":
        model = TimeSeriesTransformer(
            n_features=n_features,
            d_model=MODEL_CONFIG["transformer_d_model"],
            nhead=MODEL_CONFIG["transformer_nhead"],
            num_layers=MODEL_CONFIG["transformer_num_layers"],
            dim_feedforward=MODEL_CONFIG["transformer_dim_feedforward"],
            dropout=MODEL_CONFIG["transformer_dropout"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: linearRegression, LSTM, CNN-LSTM, TCN, SPHnet, TimeSeriesTransformer")
    
    
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Input shape: (batch, {MODEL_CONFIG['seq_len']}, {MODEL_CONFIG['inFeatures']})")
    print(f"  Output shape: (batch, {MODEL_CONFIG['outFeatures']})")
    
    # Initialize final layer bias to zero to prevent initial bias
    # This helps prevent the model from collapsing to one direction
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == MODEL_CONFIG['outFeatures']:
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                print(f"  Initialized final layer bias to zero: {name}")
    
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
    Now uses YAML configuration templates.
    """
    global DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, EVAL_CONFIG, EXPERIMENT_CONFIG
    
    try:
        # Load configuration with specified template
        config = load_config("config.yaml", template_name=experiment_name)
        
        # Update global configuration dictionaries
        update_configs_from_yaml(config)
        
        # Handle string-based scaler selection
        if isinstance(DATA_CONFIG.get("normalizeFunc"), str):
            DATA_CONFIG["normalizeFunc"] = get_scaler_from_string(DATA_CONFIG["normalizeFunc"])
        
        # Update device based on availability
        if torch.cuda.is_available():
            TRAINING_CONFIG["device"] = "cuda"
        else:
            TRAINING_CONFIG["device"] = "cpu"
        
        # Ensure model seq_len matches data window_size
        MODEL_CONFIG["seq_len"] = DATA_CONFIG["window_size"]
        
        print(f"✅ Loaded experiment template: {experiment_name}")
        
    except Exception as e:
        print(f"⚠️  Could not load template '{experiment_name}': {e}")
        print("Available templates: quick_test, full_experiment, directional_focus")
        print("Using default configuration.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    # ==================== LOAD CONFIGURATION FROM YAML ====================
    # Load configuration from config.yaml file
    
    try:
        # Load base configuration
        config = load_config("config.yaml")
        
        # Update global configuration dictionaries
        update_configs_from_yaml(config)
        
        # Handle string-based scaler selection
        if isinstance(DATA_CONFIG.get("normalizeFunc"), str):
            DATA_CONFIG["normalizeFunc"] = get_scaler_from_string(DATA_CONFIG["normalizeFunc"])
        
        # Update device based on availability
        if torch.cuda.is_available():
            TRAINING_CONFIG["device"] = "cuda"
        else:
            TRAINING_CONFIG["device"] = "cpu"
        
        # Ensure model seq_len matches data window_size
        MODEL_CONFIG["seq_len"] = DATA_CONFIG["window_size"]
        
        print("\n" + "="*60)
        print("CONFIGURATION LOADED FROM YAML")
        print("="*60)
        print(f"Model: {MODEL_CONFIG['model_type']}")
        print(f"Loss: {TRAINING_CONFIG['criterion']} (alpha={TRAINING_CONFIG['alpha']})")
        print(f"Epochs: {TRAINING_CONFIG['epochs']}")
        print(f"Device: {TRAINING_CONFIG['device']}")
        print(f"Features: {DATA_CONFIG['feature_mode']} mode")
        print(f"Scaler: {DATA_CONFIG['normalizeFunc'].__name__}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration from tests.py")
        # Fall back to default configuration if YAML loading fails
        pass

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