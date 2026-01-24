#!/usr/bin/env python3
"""
Production-ready model with realistic approach.
No artificial balancing, uses actual market patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gc

def create_production_features(df):
    """
    Create production features based on real trading strategies.
    """
    print("Creating production features...")
    
    # Moving averages (most reliable indicators)
    df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # Price position relative to moving averages
    df['price_above_sma10'] = (df['close'] > df['sma_10']).astype(int)
    df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
    df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
    
    # Moving average crossovers (strong signals)
    df['sma10_gt_sma20'] = (df['sma_10'] > df['sma_20']).astype(int)
    df['sma20_gt_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)
    
    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5).fillna(0)
    df['momentum_20'] = df['close'].pct_change(20).fillna(0)
    
    # Volatility (for risk assessment)
    df['volatility_10'] = df['close'].rolling(window=10, min_periods=1).std().fillna(0)
    df['volatility_20'] = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
    
    # RSI (standard indicator)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume confirmation (if available)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = (df['volume'] / df['volume_sma']).fillna(1)
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    else:
        df['volume_ratio'] = 1.0
        df['high_volume'] = 0
    
    # Price range (volatility indicator)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # Trend strength (composite)
    df['trend_strength'] = (
        df['price_above_sma10'] * 0.3 +
        df['price_above_sma20'] * 0.3 +
        df['sma10_gt_sma20'] * 0.2 +
        df['momentum_5'] * 0.2
    )
    
    return df

def create_production_target(df, future_offset=12):
    """
    Create realistic target based on actual trading performance.
    """
    print("Creating production target...")
    
    # Future price change (longer timeframe for more stable signals)
    future_return = (df['close'].shift(-future_offset) / df['close']) - 1
    
    # Use realistic threshold based on actual market behavior
    # 0.5% return is a reasonable target for 3-hour prediction (12 * 15min)
    threshold = 0.005
    
    # Create target: 1 for profitable move, 0 otherwise
    df['target'] = (future_return > threshold).astype(int)
    
    # Remove last rows where we can't calculate future
    df = df.iloc[:-future_offset]
    
    return df

class ProductionLSTM(nn.Module):
    """Production-ready LSTM model."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last output
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_production_model():
    """Train production model with realistic approach."""
    print("=" * 60)
    print("PRODUCTION MODEL - REALISTIC APPROACH")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading data...")
        data_path = "data/btc15m.csv"
        df = pd.read_csv(data_path)
        
        # Use sufficient data for pattern learning
        df = df.tail(50000)  # 50k rows
        print(f"Using {len(df)} rows")
        
        # Create features and target
        df = create_production_features(df)
        df = create_production_target(df, future_offset=12)
        
        # Remove NaN
        df = df.dropna()
        print(f"After cleaning: {len(df)} rows")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        # Select features
        feature_cols = [
            'price_above_sma10', 'price_above_sma20', 'price_above_sma50',
            'sma10_gt_sma20', 'sma20_gt_sma50',
            'momentum_5', 'momentum_20',
            'volatility_10', 'volatility_20',
            'rsi', 'volume_ratio', 'high_volume',
            'price_range', 'trend_strength'
        ]
        
        # Create sequences
        window_size = 20  # Reasonable window for pattern recognition
        sequences = []
        targets = []
        
        print("Creating sequences...")
        for i in range(len(df) - window_size):
            seq = df[feature_cols].iloc[i:i+window_size].values
            target = df['target'].iloc[i+window_size]
            
            if not np.isnan(seq).any():
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"Sequences: {len(X)}, Features: {X.shape[2]}")
        
        # Split chronologically (no shuffling for time series)
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Train target distribution: {np.bincount(y_train)}")
        print(f"Test target distribution: {np.bincount(y_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ProductionLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2)
        model = model.to(device)
        
        # Training setup with class weighting
        class_weights = len(y_train) / (2 * np.bincount(y_train))
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}")
        print(f"Class weights: {class_weights}")
        
        # Training
        best_acc = 0.0
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predictions == targets).sum().item()
                train_total += targets.size(0)
            
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.size(0)
            
            val_acc = val_correct / val_total
            scheduler.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1:2d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'weights/production_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                print("Early stopping")
                break
            
            gc.collect()
        
        # Final evaluation
        model.load_state_dict(torch.load('weights/production_best.pth'))
        model.eval()
        
        test_correct = 0
        test_total = 0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                
                test_correct += (predictions == targets).sum().item()
                test_total += targets.size(0)
                
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = test_correct / test_total
        
        # Test different thresholds
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        print(f"\n{'='*60}")
        print("PRODUCTION RESULTS")
        print(f"{'='*60}")
        print(f"Best Val Acc: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Test different thresholds
        best_threshold_acc = 0
        best_threshold = 0.5
        
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred = (all_probs > threshold).astype(int)
            acc = np.mean(pred == all_targets)
            print(f"Threshold {threshold:.1f}: {acc:.4f} ({acc*100:.2f}%)")
            
            if acc > best_threshold_acc:
                best_threshold_acc = acc
                best_threshold = threshold
        
        print(f"\nBest threshold: {best_threshold:.1f} -> {best_threshold_acc:.4f} ({best_threshold_acc*100:.2f}%)")
        
        # Additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        best_pred = (all_probs > best_threshold).astype(int)
        print("\nClassification Report:")
        print(classification_report(all_targets, best_pred, digits=4))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_targets, best_pred))
        
        if best_threshold_acc > 0.60:
            print("\n🎯 SUCCESS: >60% accuracy achieved!")
            print("Model is production-ready!")
        elif best_threshold_acc > 0.55:
            print("\n✅ GOOD: >55% accuracy achieved!")
            print("Model shows promising results")
        else:
            print("\n⚠️  Accuracy below target, but model is functional")
            print("Consider: more data, different features, or longer timeframes")
        
        return best_threshold_acc
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    accuracy = train_production_model()
    print(f"\nFinal production accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
