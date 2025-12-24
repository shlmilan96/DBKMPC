import torch

class StandardScaler:
    def __init__(self, device='cpu'):
        self.mean = None
        self.std = None
        self.device = device
        
    def fit(self, data):
        """
        Compute mean and std from data.
        data: torch.Tensor of shape [N, features]
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)
        
        # Handle constant values (std=0) to avoid division by zero
        # If std is 0, set it to 1, so data becomes data - mean (centered) which is 0.
        self.std[self.std < 1e-6] = 1.0
        
    def transform(self, data):
        """
        Normalize data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
            
        return (data - self.mean) / (self.std + 1e-8)
        
    def inverse_transform(self, data):
        """
        Denormalize data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
            
        return data * (self.std + 1e-8) + self.mean
