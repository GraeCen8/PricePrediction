import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ==================================================
# Feature Engineering
# ==================================================
def FeatureEngineeringExample(df, ema_n=20):
    df = df.copy()
    eps = 1e-8

    df['logRet'] = np.log(df['close'] / df['close'].shift(1))
    df['absLogRet'] = df['logRet'].abs()
    df['velocity'] = df['logRet'] - df['logRet'].shift(1)

    df['range'] = df['high'] - df['low']
    df['logRange'] = np.log(df['range'] + eps)
    df['signedLogRange'] = np.sign(df['logRet']) * df['logRange']

    prev_close = df['close'].shift(1)
    df['trueRange'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )

    df['emaTR'] = df['trueRange'].ewm(span=ema_n, adjust=False).mean()
    df['trRatio'] = df['trueRange'] / (df['emaTR'] + eps)

    df['emaAbsRet'] = df['absLogRet'].ewm(span=ema_n, adjust=False).mean()
    df['volShock'] = df['absLogRet'] - df['emaAbsRet']
    df['volPersist'] = (df['absLogRet'] > df['emaAbsRet']).astype(int)

    df['emaClose'] = df['close'].ewm(span=ema_n, adjust=False).mean()
    df['emaDist'] = (df['close'] - df['emaClose']) / (df['emaAbsRet'] + eps)
    df['reversal'] = -np.sign(df['logRet'].shift(1)) * df['logRet']

    if 'volume' in df.columns:
        df['emaVol'] = df['volume'].ewm(span=ema_n, adjust=False).mean()
        df['volSurprise'] = np.log((df['volume'] + eps) / (df['emaVol'] + eps))
        df['volPriceInteraction'] = df['logRet'] * df['volSurprise']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

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

    def __init__(
        self,
        from_file=True,
        file_loc='data',
        timeframe='15m',
        pair='btc',
        dataframe=None,
        date_column='timestamp',
        dataProcessFunc=lambda df: FeatureEngineeringExample(df, ema_n=25),
        normalizeFunc=StandardScaler,
        isClassifier=False,
        datasetClass=TimeSeriesDataset,
        batch_size=64,
        num_workers=2,
        target_column='logRet',
        window_size=128,
        split_ratio='0.7,0.2,0.1',
        shuffle=False
    ):

        self.from_file = from_file
        self.file_loc = file_loc
        self.timeframe = timeframe
        self.pair = pair
        self.dataframe = dataframe
        self.date_column = date_column
        self.dataProcessFunc = dataProcessFunc
        self.isClassifier = isClassifier
        self.datasetClass = datasetClass
        self.batch_size = batch_size
        self.normalizeFunc = normalizeFunc
        self.num_workers = num_workers
        self.target_column = target_column
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.shuffle = shuffle

    # --------------------------------------------------
    def get_data(self):
        if self.from_file:
            file_name = f"{self.pair}{self.timeframe}.csv"
            path = f"{self.file_loc}/{file_name}"
            df = pd.read_csv(path)
        elif self.dataframe is not None:
            df = self.dataframe.copy()
        else:
            raise ValueError("You must provide either a file or a dataframe")
        df = df.drop(self.date_column, axis = 1)
        return df

    # --------------------------------------------------
    def split(self, df):
        ratios = tuple(map(float, self.split_ratio.split(',')))
        assert abs(sum(ratios) - 1.0) < 1e-6

        n = len(df)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # --------------------------------------------------
    def scale(self, train_df, val_df, test_df):
        scaler = self.normalizeFunc()

        scaler.fit(train_df)
        train_scaled = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
        val_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
        test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

        return train_scaled, val_scaled, test_scaled, scaler

    # --------------------------------------------------
    def slide(self, df):
        X, y = [], []
        data = df.to_numpy()
        target_idx = df.columns.get_loc(self.target_column)

        for i in range(len(df) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size, target_idx])

        return np.array(X), np.array(y)

    # --------------------------------------------------
    def process(self):
        df = self.get_data()

        trainDF, valDF, testDF = self.split(df)

        trainDF = self.dataProcessFunc(trainDF)
        valDF = self.dataProcessFunc(valDF)
        testDF = self.dataProcessFunc(testDF)

        trainDF, valDF, testDF, scaler = self.scale(trainDF, valDF, testDF)

        trainX, trainY = self.slide(trainDF)
        valX, valY = self.slide(valDF)
        testX, testY = self.slide(testDF)

        trainDataset = self.datasetClass(trainX, trainY, self.isClassifier)
        valDataset = self.datasetClass(valX, valY, self.isClassifier)
        testDataset = self.datasetClass(testX, testY, self.isClassifier)

        trainLoader = DataLoader(trainDataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        valLoader = DataLoader(valDataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        testLoader = DataLoader(testDataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        fullDF = pd.concat([trainDF, valDF, testDF])

        for X, y in trainLoader:
            print("X:", X.shape)
            print("y:", y.shape)
            break

        return trainLoader, valLoader, testLoader, scaler, fullDF


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
    }

    processor = processing(**dataPARAMS)
    trainLoader, valLoader, testLoader, scaler, fullDF = processor.process()

   #