

import torch
import torch.nn as nn 
from funcs.process import processing, FeatureEngineeringExample, TimeSeriesDataset
from funcs.train import Training
from sklearn.preprocessing import StandardScaler
from models.linearRegression import linearRegression
import funcs.evalLoss as eval

def evalPipeline():
    data_processing_params = {
    "from_file": True,
    "file_loc": "data",
    "timeframe": "15m",
    "pair": "btc",
    "dataframe": None,
    "date_column": "timestamp",
    "dataProcessFunc": lambda df: FeatureEngineeringExample(df, ema_n=25),
    "normalizeFunc": StandardScaler,
    "isClassifier": False,
    "datasetClass": TimeSeriesDataset,
    "batch_size": 64,
    "num_workers": 2,
    "target_column": "logRet",
    "window_size": 128,
    "split_ratio": "0.7,0.2,0.1",
    "shuffle": False
}
    processor = processing(**data_processing_params)
    trainLoader, valLoader, testLoader, scaler, fullDF = processor.process()

    model = linearRegression(
        inFeatures=23,
        outFeatures=1,
        seq_len=data_processing_params['window_size'],  # correct!
        paramScale=4
)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    training_params = {
        "model": model,
        "optimizer": optimizer,
        "trainLoader": trainLoader,
        "valLoader": valLoader,
        "testLoader": testLoader,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=400,
         #   step_size=5,
            last_epoch=-1
        ),
        "epochs": 400,
        "valGap": 1,
        "saveGap": 100,
        "criterion": nn.MSELoss(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "modelType": "regressor",
    }
    trainer = Training(**training_params)
    trainer.train()

    print('training finished')

if __name__ =='__main__':
    evalPipeline()