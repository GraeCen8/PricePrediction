import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features=20, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Project features to d_model dimension
        self.feature_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len=128, features=20)
        x = self.feature_projection(x)  # (batch, 128, d_model)
        x = x + self.pos_encoder  # Add positional encoding
        x = self.transformer(x)  # (batch, 128, d_model)
        x = x[:, -1, :]  # Take last timestep
        x = self.fc_out(x)  # (batch, 1)
        return x.squeeze(-1)