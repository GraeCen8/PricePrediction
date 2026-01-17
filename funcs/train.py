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
from typing import Dict, Any, Optional

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
    def __init__(
        self,
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
        device='cuda' if torch.cuda.is_available() else 'cpu',
        modelType='regressor',
        experiment_name='experiment',
        use_tensorboard=True,
        log_dir='runs',
        log_gradients=False,
        log_weights=False,
        log_histograms=False,
        early_stopping_patience=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.trainLoader = trainLoader  # FIXED: Was incorrectly using testLoader
        self.valLoader = valLoader
        self.testLoader = testLoader
        self.scheduler = scheduler
        self.epochs = epochs
        self.valGap = valGap
        self.saveGap = saveGap
        self.criterion = criterion
        self.device = device
        self.modelType = modelType
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.log_histograms = log_histograms
        self.early_stopping_patience = early_stopping_patience
        
        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            self.writer = self._init_tensorboard()
        else:
            self.writer = None
        
        # Training metrics storage
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"Training initialized on device: {self.device}")
        if self.use_tensorboard:
            print(f"TensorBoard logging enabled. Run: tensorboard --logdir={self.log_dir}")
    
    #-----
    def _init_tensorboard(self):
        """Initialize TensorBoard SummaryWriter."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"{self.experiment_name}_{timestamp}"
        log_path = os.path.join(self.log_dir, experiment_dir)
        os.makedirs(log_path, exist_ok=True)
        
        print(f"TensorBoard logs will be saved to: {log_path}")
        return SummaryWriter(log_path)
    
    #-----
    def _log_tensorboard_scalars(self, epoch, train_loss, val_loss, lr):
        """Log scalar values to TensorBoard."""
        if self.writer is None:
            return
            
        # Log losses
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Log learning rate
        self.writer.add_scalar('Learning_rate', lr, epoch)
        
        # Log combined plot
        if val_loss is not None:
            self.writer.add_scalars('Loss_combined', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
    
    #-----
    def _log_tensorboard_histograms(self, epoch):
        """Log weight and gradient histograms to TensorBoard."""
        if self.writer is None or not (self.log_weights or self.log_gradients):
            return
            
        for name, param in self.model.named_parameters():
            if self.log_weights and param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param, epoch)
                
            if self.log_gradients and param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)
    
    #-----
    def _log_tensorboard_metrics(self, epoch, metrics: Optional[Dict[str, Any]] = None):
        """Log additional metrics to TensorBoard."""
        if self.writer is None:
            return
            
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
    
    #-----
    def _check_early_stopping(self, val_loss):
        """Check if training should stop early."""
        if self.early_stopping_patience is None:
            return False
            
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement.")
                return True
            return False
    
    #-----
    @staticmethod
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

    #-----
    def TrainTorch(self, model):
        model = model.to(self.device)
        model.train()

        # CSV setup
        log_dir = f"logs/{model.__class__.__name__}_{self.experiment_name}"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "training_log.csv")
        
        # Store metrics for end-of-training write
        metrics = []
        
        # Add model graph to TensorBoard
        if self.writer is not None:
            # Get a batch of data to visualize model graph
            try:
                sample_batch, _ = next(iter(self.trainLoader))
                sample_batch = sample_batch.to(self.device)
                self.writer.add_graph(model, sample_batch)
            except Exception as e:
                print(f"Could not add model graph to TensorBoard: {e}")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            model.train()  # Ensure model is in training mode
            
            progress_bar = tqdm.tqdm(
                self.trainLoader, 
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=True
            )

            for batch_idx, (xb, yb) in enumerate(progress_bar):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                outputs = model(xb)

                # Handle binary / regression / generic outputs
                if outputs.ndim > yb.ndim:
                    outputs = outputs.squeeze()

                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                # Log batch loss to TensorBoard (optional, can be noisy)
                if self.writer is not None and batch_idx % 10 == 0:  # Log every 10 batches
                    self.writer.add_scalar('Loss/batch', loss.item(), 
                                          epoch * len(self.trainLoader) + batch_idx)

            epoch_loss /= len(self.trainLoader)
            self.train_losses.append(epoch_loss)
            
            # ----- learning rate -----
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # ----- scheduler step -----
            if self.scheduler is not None:
                # Check if scheduler is ReduceLROnPlateau or other
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # For ReduceLROnPlateau, we need validation loss
                    pass  # Will handle after validation
                else:
                    self.scheduler.step()
            
            # ----- validation -----
            val_loss_value = None
            val_metrics = None
            if self.valLoader is not None and (epoch % self.valGap == 0 or epoch == self.epochs - 1):
                model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for xb, yb in self.valLoader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)

                        outputs = model(xb)
                        if outputs.ndim > yb.ndim:
                            outputs = outputs.squeeze()

                        loss = self.criterion(outputs, yb)
                        val_loss += loss.item()
                        
                        # Store for metrics calculation
                        val_preds.append(outputs.cpu())
                        val_targets.append(yb.cpu())

                val_loss /= len(self.valLoader)
                val_loss_value = val_loss
                self.val_losses.append(val_loss)
                
                # Calculate additional metrics for regression
                if self.modelType == 'regressor' and val_preds:
                    val_preds_tensor = torch.cat(val_preds, dim=0)
                    val_targets_tensor = torch.cat(val_targets, dim=0)
                    
                    # Calculate R² score
                    from sklearn.metrics import r2_score
                    r2 = r2_score(val_targets_tensor.numpy(), val_preds_tensor.numpy())
                    val_metrics = {'r2_score': r2}
                
                model.train()  # Back to training mode
                
                # Check early stopping
                if self._check_early_stopping(val_loss_value):
                    self.early_stop = True
                
                # Handle ReduceLROnPlateau scheduler during training
                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss_value)
            
            # ----- TensorBoard logging -----
            if self.writer is not None:
                self._log_tensorboard_scalars(epoch, epoch_loss, val_loss_value, current_lr)
                self._log_tensorboard_histograms(epoch)
                if val_metrics:
                    self._log_tensorboard_metrics(epoch, val_metrics)
            
            # Store metrics for CSV
            metrics.append({
                'epoch': epoch,
                'train_loss': epoch_loss,
                'val_loss': val_loss_value if val_loss_value is not None else '',
                'learning_rate': current_lr,
                'r2_score': val_metrics.get('r2_score', '') if val_metrics else ''
            })
            #6dp
            valOut = str(val_loss_value.__round__(6)) if val_loss_value else 'N/A'
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.epochs}: "
                  f"Train Loss: {epoch_loss:.6f}, "
                  f"Val Loss: {valOut}, "
                  f"LR: {current_lr:.6f}")
            
            # ----- checkpoint -----
            # Convert saveGap to int if it's a float (for backward compatibility)
            saveGap_int = int(self.saveGap) if isinstance(self.saveGap, float) else self.saveGap
            if (isinstance(saveGap_int, int) and (epoch % saveGap_int == 0)) or epoch == self.epochs - 1:
                os.makedirs("weights", exist_ok=True)
                modelName = f"weights/checkpoint_{self.experiment_name}_epoch{epoch+1}_loss{epoch_loss:.4f}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': epoch_loss,
                    'val_loss': val_loss_value,
                    'learning_rate': current_lr
                }, modelName)
                print(f"Checkpoint saved: {modelName}")
            
            # Break if early stopping triggered
            if self.early_stop:
                print(f"Training stopped early at epoch {epoch+1}")
                break
        
        # ----- scheduler step for ReduceLROnPlateau (after validation) -----
        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss_value is not None:
                self.scheduler.step(val_loss_value)
        
        # Write all metrics to CSV at the end
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'r2_score'])
            for metric in metrics:
                writer.writerow([
                    metric['epoch'], 
                    metric['train_loss'], 
                    metric['val_loss'], 
                    metric['learning_rate'],
                    metric['r2_score']
                ])
        
        # Save final model
        os.makedirs("weights", exist_ok=True)
        final_val_loss = val_loss_value if val_loss_value else self.train_losses[-1]
        modelName = f"weights/{self.experiment_name}_final_val{final_val_loss:.4f}.pth"
        torch.save(model.state_dict(), modelName)
        print(f"Final model saved: {modelName}")
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            
    #-----
    def train(self):
        modeltype = detect_model_type(self.model)
        if modeltype == 'torch':
            self.TrainTorch(self.model)
        elif modeltype == 'sklearn':
            self.TrainSk(self.model)
        else:
            print('Unsupported model type: only sklearn and torch available')
            raise ValueError("Unsupported model type")
            
        # Plot training curves if matplotlib is available
        try:
            self.plot_training_curves()
        except:
            pass
    
    #-----
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        if not self.train_losses:
            return
            
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        if self.val_losses:
            # Align validation losses with training epochs
            val_epochs = [i for i in range(len(self.train_losses)) if i % self.valGap == 0 or i == len(self.train_losses)-1]
            val_epochs = val_epochs[:len(self.val_losses)]  # Ensure same length
            plt.plot(val_epochs, self.val_losses, label='Val Loss', linewidth=2, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.experiment_name} - Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates, label='Learning Rate', color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("plots", exist_ok=True)
        plot_path = f"plots/{self.experiment_name}_training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")