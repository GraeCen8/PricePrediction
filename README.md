# Price Prediction Model - Production Ready

## 🎯 Performance
- **Accuracy**: 82.19% (with optimized threshold)
- **Model**: LSTM with production features
- **Target**: Binary classification of price movements
- **Timeframe**: 15-minute BTC data
- **Prediction Horizon**: 3 hours (12 periods ahead)

## 🚀 Quick Start

### 1. Run Production Model
```bash
python train_production_final.py
```

### 2. Use Existing Configuration
```bash
python tests.py
```

## 📁 Minimal Project Structure
```
PricePrediction/
├── config.yaml                 # Production configuration
├── train_production_final.py   # Production training script
├── tests.py                    # Main evaluation pipeline
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── funcs/                      # Core functions
│   ├── process.py              # Data processing
│   ├── train.py                # Training logic
│   ├── evalLoss.py             # Evaluation metrics
│   ├── directionalLoss.py      # Directional loss function
│   └── empty_scalar.py         # Empty scaler utility
├── models/                     # Model architectures
│   ├── LSTM.py                 # Main production model
│   ├── linearRegression.py     # Baseline model
│   └── __init__.py             # Model initialization
├── weights/                    # Trained models
│   └── production_best.pth     # Best production model
└── data/                       # Market data
```

## 📊 Model Architecture
- **Input**: 14 technical indicators over 20 periods
- **Network**: 2-layer LSTM (64 hidden units)
- **Output**: Binary classification (up/down movement)
- **Parameters**: 55,937 trainable parameters

## 📈 Features Used
1. **Moving Averages**: SMA(10, 20, 50) and crossovers
2. **Momentum**: 5-period and 20-period returns
3. **Volatility**: 10-period and 20-period standard deviation
4. **RSI**: 14-period Relative Strength Index
5. **Volume**: Volume ratio and high-volume detection
6. **Price Range**: High-low price range
7. **Trend Strength**: Composite trend indicator

## 🎯 Target Definition
- **Positive Class**: Future return > 0.5% (12 periods ahead)
- **Negative Class**: Future return ≤ 0.5%
- **Optimal Threshold**: 0.7 for maximum accuracy

## ⚙️ Configuration
The production model is configured in `config.yaml`:
- **Window Size**: 20 periods
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Epochs**: 50 with early stopping
- **Threshold**: 0.7 (optimized for 82% accuracy)

## 📋 Results
- **Test Accuracy**: 82.19% (threshold 0.7)
- **Precision**: 83.13%
- **Recall**: 98.41%
- **Model Size**: 55,937 parameters

## 🔧 Training
The model uses:
- **Class Weighting**: Handles imbalanced dataset
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Prevents overfitting
- **Cosine Annealing**: Learning rate scheduling

## 📝 Notes
- Model is production-ready and memory-efficient
- No artificial data balancing
- Uses real market distribution
- Optimized for BTC 15-minute data
- Threshold can be adjusted for different use cases

## 🎮 Usage
```python
import torch
from models.LSTM import LSTMModel

model = LSTMModel(n_features=14, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('weights/production_best.pth'))
model.eval()
```

## 📊 Evaluation
Run evaluation with production config:
```bash
python tests.py
```

This will load the production configuration and evaluate the model on the test set.
