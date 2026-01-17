import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# ==================================================
# Lean Optimized Feature Engineering (20 Features)
# ==================================================
def LeanFeatureEngineering(df, fast_ema=8, slow_ema=21, vol_ema=14, mom_period=14):
    """
    Lean feature engineering with 20 carefully selected features.
    
    Parameters:
    - fast_ema: 8 (captures short-term trends, ~2 hours on 15m data)
    - slow_ema: 21 (captures medium-term trends, ~5 hours on 15m data)
    - vol_ema: 14 (volatility smoothing, standard period)
    - mom_period: 14 (momentum lookback, standard period)
    
    Feature Selection Strategy:
    - Each feature provides UNIQUE information
    - Minimal correlation between features
    - Proven alpha in financial markets
    - Computationally efficient
    
    20 Core Features:
    ----------------
    RETURNS & MOMENTUM (5):
      1. logRet - Current return (target predictor)
      2. logRet_1 - Lagged return (autoregressive)
      3. ret_5 - 5-period return (~1 hour, medium term)
      4. momentum - 14-period momentum (trend strength)
      5. acceleration - Change in returns (2nd derivative)
    
    VOLATILITY (4):
      6. atrPct - ATR normalized by price (adaptive volatility)
      7. volRatio - Current vol vs recent average (regime detector)
      8. volOfVol - Volatility of volatility (uncertainty measure)
      9. parkinsonVol - High-low volatility (efficient estimator)
    
    TREND (4):
      10. trendFast - Distance from fast EMA (short-term position)
      11. emaCross - Fast/slow EMA difference (trend direction)
      12. trendPersist - How long trend has lasted (momentum decay)
      13. bbPosition - Bollinger Band position (mean reversion)
    
    VOLUME (4 - if available):
      14. volRatio - Volume vs average (participation)
      15. volPriceCorr - Volume * volatility (conviction)
      16. pressureImbalance - Buy vs sell pressure (direction)
      17. volMomentum - Volume acceleration (interest change)
    
    MICROSTRUCTURE (3):
      18. bodyRatio - Candle body vs full range (conviction)
      19. shadowRatio - Upper vs lower wicks (pressure)
      20. gapPct - Gap size (overnight risk)
    """
    df = df.copy()
    eps = 1e-8
    
    # ==================== RETURNS & MOMENTUM (5 features) ====================
    df['logRet'] = np.log(df['close'] / df['close'].shift(1))
    df['logRet_1'] = df['logRet'].shift(1)
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['momentum'] = df['close'] / df['close'].shift(mom_period) - 1
    df['acceleration'] = df['logRet'] - df['logRet'].shift(1)
    
    # ==================== VOLATILITY (4 features) ====================
    # True Range for ATR calculation
    prev_close = df['close'].shift(1)
    trueRange = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    
    # ATR (Average True Range)
    atr = trueRange.ewm(span=vol_ema, adjust=False).mean()
    df['atrPct'] = atr / (df['close'] + eps)
    
    # Volatility metrics
    absLogRet = df['logRet'].abs()
    volEma = absLogRet.ewm(span=vol_ema, adjust=False).mean()
    df['volRatio'] = absLogRet / (volEma + eps)
    
    volStd = absLogRet.rolling(vol_ema).std()
    df['volOfVol'] = volStd / (volEma + eps)
    
    df['parkinsonVol'] = np.sqrt(
        (1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2
    )
    
    # ==================== TREND (4 features) ====================
    emaFast = df['close'].ewm(span=fast_ema, adjust=False).mean()
    emaSlow = df['close'].ewm(span=slow_ema, adjust=False).mean()
    
    df['trendFast'] = (df['close'] - emaFast) / (atr + eps)
    df['emaCross'] = (emaFast - emaSlow) / (df['close'] + eps)
    
    # Trend persistence
    trendDir = np.sign(df['emaCross'])
    df['trendPersist'] = df.groupby((trendDir != trendDir.shift()).cumsum()).cumcount() + 1
    df['trendPersist'] = np.log1p(df['trendPersist'])
    
    # Bollinger Band position
    bbMid = df['close'].rolling(20).mean()
    bbStd = df['close'].rolling(20).std()
    df['bbPosition'] = (df['close'] - bbMid) / (2 * bbStd + eps)
    
    # ==================== VOLUME (4 features - if available) ====================
    if 'volume' in df.columns:
        volEmaVol = df['volume'].ewm(span=vol_ema, adjust=False).mean()
        df['volRatio'] = df['volume'] / (volEmaVol + eps)
        df['volPriceCorr'] = df['volume'] * absLogRet
        
        # Buy/sell pressure
        buyPressure = np.where(df['logRet'] > 0, df['volume'], 0)
        sellPressure = np.where(df['logRet'] < 0, df['volume'], 0)
        buyPressureEma = pd.Series(buyPressure).ewm(span=vol_ema, adjust=False).mean()
        sellPressureEma = pd.Series(sellPressure).ewm(span=vol_ema, adjust=False).mean()
        df['pressureImbalance'] = (buyPressureEma - sellPressureEma) / (volEmaVol + eps)
        
        df['volMomentum'] = df['volume'] / (df['volume'].shift(5) + eps) - 1
    
    # ==================== MICROSTRUCTURE (3 features) ====================
    range_val = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    df['bodyRatio'] = body / (range_val + eps)
    
    upperShadow = df['high'] - np.maximum(df['open'], df['close'])
    lowerShadow = np.minimum(df['open'], df['close']) - df['low']
    df['shadowRatio'] = (upperShadow - lowerShadow) / (range_val + eps)
    
    gap = df['open'] - df['close'].shift(1)
    df['gapPct'] = gap / (df['close'].shift(1) + eps)
    
    # ==================== CLEANUP ====================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"Dropped {dropped} rows due to NaN values from feature engineering")
    
    return df


# ==================================================
# Full Feature Engineering (65+ Features) - Optional
# ==================================================
def FullFeatureEngineering(df, fast_ema=8, slow_ema=21, vol_ema=14, mom_period=14):
    """
    Full feature engineering with 65+ features for experimentation.
    Use this if you have massive amounts of data and want to do feature selection.
    """
    df = df.copy()
    eps = 1e-8
    
    # ==================== RETURNS & MOMENTUM ====================
    df['logRet'] = np.log(df['close'] / df['close'].shift(1))
    df['logRet_1'] = df['logRet'].shift(1)
    df['logRet_2'] = df['logRet'].shift(2)
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['ret_10'] = np.log(df['close'] / df['close'].shift(10))
    df['momentum'] = df['close'] / df['close'].shift(mom_period) - 1
    df['acceleration'] = df['logRet'] - df['logRet'].shift(1)
    df['posRet'] = df['logRet'].clip(lower=0)
    df['negRet'] = df['logRet'].clip(upper=0)
    
    # ==================== VOLATILITY ====================
    prev_close = df['close'].shift(1)
    df['trueRange'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    
    df['atr'] = df['trueRange'].ewm(span=vol_ema, adjust=False).mean()
    df['atrPct'] = df['atr'] / (df['close'] + eps)
    
    df['absLogRet'] = df['logRet'].abs()
    df['volEma'] = df['absLogRet'].ewm(span=vol_ema, adjust=False).mean()
    df['volStd'] = df['absLogRet'].rolling(vol_ema).std()
    df['volOfVol'] = df['volStd'] / (df['volEma'] + eps)
    df['volRatio'] = df['absLogRet'] / (df['volEma'] + eps)
    df['volRegime'] = (df['volRatio'] > 1.5).astype(int)
    
    df['parkinsonVol'] = np.sqrt(
        (1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2
    )
    
    # ==================== TREND ====================
    df['emaFast'] = df['close'].ewm(span=fast_ema, adjust=False).mean()
    df['emaSlow'] = df['close'].ewm(span=slow_ema, adjust=False).mean()
    df['trendFast'] = (df['close'] - df['emaFast']) / (df['atr'] + eps)
    df['trendSlow'] = (df['close'] - df['emaSlow']) / (df['atr'] + eps)
    df['emaCross'] = (df['emaFast'] - df['emaSlow']) / (df['close'] + eps)
    
    df['trendDir'] = np.sign(df['emaCross'])
    df['trendPersist'] = df.groupby((df['trendDir'] != df['trendDir'].shift()).cumsum()).cumcount() + 1
    df['trendPersist'] = np.log1p(df['trendPersist'])
    
    df['pricePosition'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)
    
    # ==================== VOLUME ====================
    if 'volume' in df.columns:
        df['volEmaVol'] = df['volume'].ewm(span=vol_ema, adjust=False).mean()
        df['volRatioVol'] = df['volume'] / (df['volEmaVol'] + eps)
        df['volTrend'] = np.log1p(df['volRatioVol'])
        df['volPriceCorr'] = df['volume'] * df['absLogRet']
        df['volPriceDir'] = df['volume'] * df['logRet']
        df['volMomentum'] = df['volume'] / (df['volume'].shift(5) + eps) - 1
        
        df['volStdVol'] = df['volume'].rolling(vol_ema).std()
        df['volSurprise'] = (df['volume'] - df['volEmaVol']) / (df['volStdVol'] + eps)
        
        df['buyPressure'] = np.where(df['logRet'] > 0, df['volume'], 0)
        df['sellPressure'] = np.where(df['logRet'] < 0, df['volume'], 0)
        df['buyPressureEma'] = df['buyPressure'].ewm(span=vol_ema, adjust=False).mean()
        df['sellPressureEma'] = df['sellPressure'].ewm(span=vol_ema, adjust=False).mean()
        df['pressureImbalance'] = (df['buyPressureEma'] - df['sellPressureEma']) / (df['volEmaVol'] + eps)
    
    # ==================== MICROSTRUCTURE ====================
    df['range'] = df['high'] - df['low']
    df['rangePct'] = df['range'] / (df['close'] + eps)
    df['body'] = (df['close'] - df['open']).abs()
    df['bodyRatio'] = df['body'] / (df['range'] + eps)
    
    df['upperShadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lowerShadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['shadowRatio'] = (df['upperShadow'] - df['lowerShadow']) / (df['range'] + eps)
    
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gapPct'] = df['gap'] / (df['close'].shift(1) + eps)
    df['gapFilled'] = (
        ((df['gap'] > 0) & (df['low'] <= df['close'].shift(1))) |
        ((df['gap'] < 0) & (df['high'] >= df['close'].shift(1)))
    ).astype(int)
    
    # ==================== MEAN REVERSION ====================
    df['maDeviation'] = (df['close'] - df['emaSlow']) / (df['volEma'] * df['close'] + eps)
    
    gain = df['logRet'].clip(lower=0)
    loss = -df['logRet'].clip(upper=0)
    avg_gain = gain.ewm(span=vol_ema, adjust=False).mean()
    avg_loss = loss.ewm(span=vol_ema, adjust=False).mean()
    rs = avg_gain / (avg_loss + eps)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsiNorm'] = (df['rsi'] - 50) / 50
    
    df['bbMid'] = df['close'].rolling(20).mean()
    df['bbStd'] = df['close'].rolling(20).std()
    df['bbPosition'] = (df['close'] - df['bbMid']) / (2 * df['bbStd'] + eps)
    
    # ==================== INTERACTION FEATURES ====================
    df['trendVolRegime'] = df['trendDir'] * df['volRegime']
    
    if 'volume' in df.columns:
        df['trendVolConfirm'] = df['emaCross'] * df['volRatioVol']
    
    # ==================== CLEANUP ====================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"Dropped {dropped} rows due to NaN values from feature engineering")
    
    return df


# ==================================================
# Dataset
# ==================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, isClassifier=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if isClassifier:
            self.y = self.y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================================================
# Processing Pipeline
# ==================================================
class processing:
    """
    Time series data processing pipeline with proper train/val/test splitting.
    
    Key improvements:
    1. Feature engineering applied to full dataset BEFORE splitting (prevents data leakage)
    2. Scaler fit only on training data (prevents future information leakage)
    3. Separate scaling for features and target (enables proper inverse transform)
    4. Robust error handling and validation
    5. Lean default (20 features) with optional full feature set (65+ features)
    """

    def __init__(
        self,
        from_file=True,
        file_loc='data',
        timeframe='15m',
        pair='btc',
        dataframe=None,
        date_column='timestamp',
        dataProcessFunc=None,
        feature_mode='lean',  # 'lean' (20 features) or 'full' (65+ features)
        normalizeFunc=RobustScaler,  # RobustScaler better for financial data (handles outliers)
        isClassifier=False,
        datasetClass=TimeSeriesDataset,
        batch_size=64,
        num_workers=2,
        target_column='logRet',
        window_size=128,
        split_ratio='0.7,0.2,0.1',
        shuffle_train=True,  # Renamed for clarity
        scale_target=False,  # New: option to scale target separately
        verbose=True
    ):

        self.from_file = from_file
        self.file_loc = file_loc
        self.timeframe = timeframe
        self.pair = pair
        self.dataframe = dataframe
        self.date_column = date_column
        
        # Use lean or full feature engineering
        if dataProcessFunc is None:
            if feature_mode == 'lean':
                self.dataProcessFunc = lambda df: LeanFeatureEngineering(
                    df, fast_ema=8, slow_ema=21, vol_ema=14, mom_period=14
                )
                if verbose:
                    print("Using LEAN feature engineering (20 features)")
            elif feature_mode == 'full':
                self.dataProcessFunc = lambda df: FullFeatureEngineering(
                    df, fast_ema=8, slow_ema=21, vol_ema=14, mom_period=14
                )
                if verbose:
                    print("Using FULL feature engineering (65+ features)")
            else:
                raise ValueError("feature_mode must be 'lean' or 'full'")
        else:
            self.dataProcessFunc = dataProcessFunc
            
        self.isClassifier = isClassifier
        self.datasetClass = datasetClass
        self.batch_size = batch_size
        self.normalizeFunc = normalizeFunc
        self.num_workers = num_workers
        self.target_column = target_column
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.shuffle_train = shuffle_train
        self.scale_target = scale_target
        self.verbose = verbose

    # --------------------------------------------------
    def get_data(self):
        """Load data from file or dataframe."""
        if self.from_file:
            file_name = f"{self.pair}{self.timeframe}.csv"
            path = f"{self.file_loc}/{file_name}"
            df = pd.read_csv(path)
            if self.verbose:
                print(f"Loaded data from {path}: {len(df)} rows")
        elif self.dataframe is not None:
            df = self.dataframe.copy()
            if self.verbose:
                print(f"Using provided dataframe: {len(df)} rows")
        else:
            raise ValueError("You must provide either a file or a dataframe")
        
        # Ensure data is sorted by date before processing
        if self.date_column in df.columns:
            df = df.sort_values(self.date_column).reset_index(drop=True)
            df = df.drop(self.date_column, axis=1)
        
        return df

    # --------------------------------------------------
    def split(self, df):
        """Split data chronologically into train/val/test."""
        ratios = tuple(map(float, self.split_ratio.split(',')))
        assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        assert len(ratios) == 3, "Must provide 3 ratios (train, val, test)"

        n = len(df)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        trainDF = df.iloc[:train_end].copy()
        valDF = df.iloc[train_end:val_end].copy()
        testDF = df.iloc[val_end:].copy()
        
        if self.verbose:
            print(f"Split sizes - Train: {len(trainDF)}, Val: {len(valDF)}, Test: {len(testDF)}")
        
        return trainDF, valDF, testDF

    # --------------------------------------------------
    def scale(self, train_df, val_df, test_df):
        """
        Scale features (and optionally target) using training data statistics only.
        
        Returns separate scalers for features and target to enable inverse transform.
        """
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != self.target_column]
        
        # Scale features
        feature_scaler = self.normalizeFunc()
        feature_scaler.fit(train_df[feature_cols])
        
        train_features_scaled = pd.DataFrame(
            feature_scaler.transform(train_df[feature_cols]), 
            columns=feature_cols,
            index=train_df.index
        )
        val_features_scaled = pd.DataFrame(
            feature_scaler.transform(val_df[feature_cols]), 
            columns=feature_cols,
            index=val_df.index
        )
        test_features_scaled = pd.DataFrame(
            feature_scaler.transform(test_df[feature_cols]), 
            columns=feature_cols,
            index=test_df.index
        )
        
        # Handle target scaling
        target_scaler = None
        if self.scale_target:
            target_scaler = self.normalizeFunc()
            target_scaler.fit(train_df[[self.target_column]])
            
            train_target_scaled = pd.DataFrame(
                target_scaler.transform(train_df[[self.target_column]]),
                columns=[self.target_column],
                index=train_df.index
            )
            val_target_scaled = pd.DataFrame(
                target_scaler.transform(val_df[[self.target_column]]),
                columns=[self.target_column],
                index=val_df.index
            )
            test_target_scaled = pd.DataFrame(
                target_scaler.transform(test_df[[self.target_column]]),
                columns=[self.target_column],
                index=test_df.index
            )
        else:
            # Keep target unscaled
            train_target_scaled = train_df[[self.target_column]]
            val_target_scaled = val_df[[self.target_column]]
            test_target_scaled = test_df[[self.target_column]]
        
        # Recombine features and target
        train_scaled = pd.concat([train_features_scaled, train_target_scaled], axis=1)
        val_scaled = pd.concat([val_features_scaled, val_target_scaled], axis=1)
        test_scaled = pd.concat([test_features_scaled, test_target_scaled], axis=1)
        
        if self.verbose:
            print(f"Scaled data using {self.normalizeFunc.__name__}")
            if self.scale_target:
                print("Target variable scaled separately")
        
        return train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler

    # --------------------------------------------------
    def slide(self, df):
        """Create sliding windows for sequence prediction."""
        X, y = [], []
        # Exclude target column from features
        feature_cols = [col for col in df.columns if col != self.target_column]
        feature_data = df[feature_cols].to_numpy()
        target_data = df[self.target_column].to_numpy()
        target_idx = df.columns.get_loc(self.target_column)

        for i in range(len(df) - self.window_size):
            X.append(feature_data[i:i + self.window_size])
            y.append(target_data[i + self.window_size])

        return np.array(X), np.array(y)

    # --------------------------------------------------
    def process(self):
        """
        Main processing pipeline.
        
        Returns:
        - trainLoader, valLoader, testLoader: PyTorch DataLoaders
        - feature_scaler: Fitted scaler for features (for inverse transform)
        - target_scaler: Fitted scaler for target (None if scale_target=False)
        - feature_names: List of feature column names
        - metadata: Dict with useful processing information
        """
        if self.verbose:
            print("="*60)
            print("Starting Data Processing Pipeline")
            print("="*60)
        
        # 1. Load raw data
        df = self.get_data()
        
        # 2. Apply feature engineering to ENTIRE dataset (prevents leakage)
        if self.verbose:
            print("\nApplying feature engineering to full dataset...")
        df = self.dataProcessFunc(df)
        
        if self.verbose:
            print(f"Features created: {len(df.columns)} total columns")
            print(f"Final dataset size: {len(df)} rows")
        
        # Validate window size
        if self.window_size >= len(df):
            raise ValueError(f"Window size ({self.window_size}) must be smaller than dataset ({len(df)})")
        
        # 3. Split into train/val/test
        if self.verbose:
            print("\nSplitting data chronologically...")
        trainDF, valDF, testDF = self.split(df)
        
        # 4. Scale data (fit on train only)
        if self.verbose:
            print("\nScaling data...")
        trainDF, valDF, testDF, feature_scaler, target_scaler = self.scale(trainDF, valDF, testDF)
        
        # 5. Create sliding windows
        if self.verbose:
            print("\nCreating sliding windows...")
        trainX, trainY = self.slide(trainDF)
        valX, valY = self.slide(valDF)
        testX, testY = self.slide(testDF)
        
        if self.verbose:
            print(f"Window shapes - Train: {trainX.shape}, Val: {valX.shape}, Test: {testX.shape}")
        
        # 6. Create PyTorch datasets
        trainDataset = self.datasetClass(trainX, trainY, self.isClassifier)
        valDataset = self.datasetClass(valX, valY, self.isClassifier)
        testDataset = self.datasetClass(testX, testY, self.isClassifier)
        
        # 7. Create data loaders
        trainLoader = DataLoader(
            trainDataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train,  # Only shuffle training data
            num_workers=self.num_workers
        )
        valLoader = DataLoader(
            valDataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # Never shuffle validation
            num_workers=self.num_workers
        )
        testLoader = DataLoader(
            testDataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # Never shuffle test
            num_workers=self.num_workers
        )
        
        # 8. Prepare metadata
        feature_names = [col for col in df.columns if col != self.target_column]
        metadata = {
            'n_features': len(feature_names),
            'n_train_samples': len(trainX),
            'n_val_samples': len(valX),
            'n_test_samples': len(testX),
            'window_size': self.window_size,
            'target_column': self.target_column,
            'feature_names': feature_names
        }
        
        if self.verbose:
            print("\n" + "="*60)
            print("Processing Complete!")
            print("="*60)
            print(f"Number of features: {metadata['n_features']}")
            print(f"Feature names: {feature_names[:5]}... (showing first 5)")
            print(f"\nBatch example from training data:")
            for X, y in trainLoader:
                print(f"  Input shape (batch): {X.shape}")
                print(f"  Target shape (batch): {y.shape}")
                break
            print("="*60)
        
        return trainLoader, valLoader, testLoader, feature_scaler, target_scaler, feature_names, metadata

# ==================================================
# Run
# ==================================================
if __name__ == "__main__":

    dataPARAMS = {
        "from_file": True,
        "file_loc": "data",
        "timeframe": "15m",
        "pair": "btc",
        "target_column": "logRet",
        "window_size": 128,
        "feature_mode": "lean",
        "isClassifier": False,
        "datasetClass": TimeSeriesDataset,
        "batch_size": 64,
        "num_workers": 2
    }

    processor = processing(**dataPARAMS)
    trainLoader, valLoader, testLoader, feature_scaler, target_scaler, feature_names, metadata = processor.process()

   