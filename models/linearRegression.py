import torch
import torch.nn as nn

class linearRegression(nn.Module): 
    def __init__(self, inFeatures, outFeatures, seq_len, paramScale=1):
        super(linearRegression, self).__init__()
        
        # Flattened input size
        input_size = inFeatures * seq_len
        
        # MLP layers
        self.L1 = nn.Linear(input_size, 128 * paramScale, bias=False)
        self.L2 = nn.Linear(128 * paramScale, 256 * paramScale, bias=False)
        self.L3 = nn.Linear(256 * paramScale, 128 * paramScale, bias=False)
        
        self.gelu = nn.GELU()
        
        # Final output
        self.out = nn.Linear(128 * paramScale, outFeatures, bias=False)
        
    def forward(self, X):
        # Flatten sequence and features
        X = X.view(X.size(0), -1)  # (batch_size, seq_len * inFeatures)
        
        # Forward pass through MLP
        X = self.L1(X)
        X = self.gelu(X)
        X = self.L2(X)
        X = self.gelu(X)
        X = self.L3(X)
        X = self.gelu(X)
        X = self.out(X)
        return X


if __name__ == '__main__':
    # Config
    config = {
        'inFeatures': 23,
        'outFeatures': 1,
        'seq_len': 64,
        'paramScale': 1
    }

    model = linearRegression(**config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dummy input
    batch_size = 32
    dummy_input = torch.randn(batch_size, config['seq_len'], config['inFeatures']).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, config['outFeatures'])
    print("✓ Output shape is correct!")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
