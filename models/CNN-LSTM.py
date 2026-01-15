import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, n_features=20, cnn_channels=64, lstm_hidden=128, dropout=0.2):
        super().__init__()
        
        # CNN extracts local patterns
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM captures temporal dependencies
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, num_layers=2, 
                           batch_first=True, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: (batch, 128, 20)
        x = x.transpose(1, 2)  # (batch, 20, 128)
        
        # CNN feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # (batch, 64, 64)
        
        x = x.transpose(1, 2)  # (batch, 64, 64)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)