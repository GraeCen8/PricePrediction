#this is meant to be a function that can detect if the model is a pytorch model or a skikit learn one, if torch then train with that else 
# then use model.fit() to train the model. make sure to make sure that it works well with binary and regression models for both and and train 
# any model i can use during the ablation tests 

import torch 
import torchvision
import math as m
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import sklearn.base 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import csv

def detect_model_type(model):
    """
    Returns: 'torch', 'sklearn', or 'unknown'
    """
    # ---- PyTorch ----
    if isinstance(model, torch.nn.Module):
        return "torch"

    # ---- scikit-learn ----
    if isinstance(model, sklearn.base.BaseEstimator):
        return "sklearn"

    return "unknown"


class Training:
    def __init__(self,
            model,
            optimizer,
            trainLoader,
            valLoader,
            testLoader,
            scheduler,
            epochs,
            valGap,
            saveGap,
            criterion,
            device = 'cuda' if torch.cuda.is_available else 'cpu',
            modelType = 'regressor'
    ):
        self.model = model
        self.optimizer = optimizer
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.testLoader = testLoader
        self.scheduler = scheduler
        self.epochs = epochs
        self.valGap = valGap
        self.saveGap = saveGap
        self.criterion = criterion
        self.device = device

    #-----
    def detect_model_type(self, model):
        """
        Returns: 'torch', 'sklearn', or 'unknown'
        """
        # ---- PyTorch ----
        if isinstance(model, torch.nn.Module):
            return "torch"

        # ---- scikit-learn ----
        if isinstance(model, sklearn.base.BaseEstimator):
            return "sklearn"

        return "unknown"
    
    #-----
    def dataloader_to_numpy(dataloader):
        X, y = [], []

        for xb, yb in dataloader:
            X.append(xb.numpy())
            y.append(yb.numpy())

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        return X, y


    #-----
    def TrainSk(self, model):
        trainX, trainY = self.dataloader_to_numpy(self.trainLoader)
        model.fit(trainX, trainY)

       # valX, valY = self.dataloader_to_numpy(self.valLoader)
       # model.predict(valX)

    #-----
    def TrainTorch(self, model):
        model = model.to(self.device)
        model.train()

        # CSV setup
        log_dir = f"logs/{model.__class__.__name__}"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "training_log.csv")
        
        # Store metrics for end-of-training write
        metrics = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for xb, yb in tqdm.tqdm(self.trainLoader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()

                outputs = model(xb)

                # handle binary / regression / generic outputs
                if outputs.ndim > yb.ndim:
                    outputs = outputs.squeeze()

                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.trainLoader)

            # ----- validation -----
            val_loss_value = None
            if self.valLoader is not None and epoch % self.valGap == 0:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for xb, yb in self.valLoader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)

                        outputs = model(xb)
                        if outputs.ndim > yb.ndim:
                            outputs = outputs.squeeze()

                        loss = self.criterion(outputs, yb)
                        val_loss += loss.item()

                val_loss /= len(self.valLoader)
                val_loss_value = val_loss
                model.train()

            # Store metrics for this epoch
            metrics.append({
                'epoch': epoch,
                'train_loss': epoch_loss,
                'val_loss': val_loss_value if val_loss_value is not None else ''
            })

            # ----- checkpoint -----
            if epoch % self.saveGap == 0:
                os.makedirs("weights", exist_ok=True)
                modelName = f"weights/checkpoint-classifierTest-v1-{epoch_loss:.2f}.pth"
                torch.save(model.state_dict(), modelName)

        # Write all metrics to CSV at the end
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
            for metric in metrics:
                writer.writerow([metric['epoch'], metric['train_loss'], metric['val_loss']])

        os.makedirs("weights", exist_ok=True)
        modelName = f"weights/classifierTest-v1-{val_loss_value:.2f}.pth"
        torch.save(model.state_dict(), modelName)
            
    #-----
    def train(self):
        modeltype = detect_model_type(self.model)
        if modeltype == 'torch':
            self.TrainTorch(self.model)
        elif modeltype == 'sklearn':
            self.TrainSk(self.model)
        else:
            print('unsuported model type: only sklearn and torch available')
            raise BrokenPipeError


#--------------------------------
#--------------------------------