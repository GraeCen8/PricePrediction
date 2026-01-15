

import torch
import torch.nn as nn 
from funcs.process import processing, FeatureEngineeringExample, TimeSeriesDataset
from funcs.train import Training
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.linearRegression import linearRegression
import funcs.evalLoss as eval
import funcs.evalLoss as eval
import funcs.directionalLoss as d
import pandas as pd
from funcs.empty_scalar import EmptyScaler

def evalPipeline():
    data_processing_params = {
    "from_file": True,
    "file_loc": "data",
    "timeframe": "15m",
    "pair": "btc",
    "dataframe": None,
    "date_column": "timestamp",
    "dataProcessFunc": lambda df: FeatureEngineeringExample(df, ema_n=25),
    "normalizeFunc": EmptyScaler,
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

    print(fullDF.head())

    model = linearRegression(
        inFeatures=23,
        outFeatures=1,
        seq_len=data_processing_params['window_size'],  # correct!
        paramScale=5
)
    optimizer = torch.optim.Muon(model.parameters(), lr=0.0001)
    training_params = {
        "model": model,
        "optimizer": optimizer,
        "trainLoader": trainLoader,
        "valLoader": valLoader,
        "testLoader": testLoader,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
         #   step_size=5,
            last_epoch=-1
        ),
        "epochs": 40,
        "valGap": 1,
        "saveGap": 0.1,
        "criterion": nn.MSELoss(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "modelType": "regressor",
    }
    trainer = Training(**training_params)
    trainer.train()

    print("evaluating ")
    y_actual, y_pred = eval.predict(model, valLoader, device='cuda')
    metrics = eval.eval_regressor_performance(
        y_actual=y_actual,
        y_pred=y_pred,
        target_name='logRet',
        threshold=0  # Use 0 for mean-centered returns
    )
    eval.print_metrics(metrics)

    print('training finished')

if __name__ =='__main__':
    evalPipeline()
