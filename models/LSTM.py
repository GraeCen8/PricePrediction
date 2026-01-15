import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, n_features=20, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, 128, 20)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last hidden state
        out = self.fc(h_n[-1])  # (batch, 1)
        return out.squeeze(-1)