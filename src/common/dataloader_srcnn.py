import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .normalization import normalize_batch, denormalize_batch

class FlowFieldDataset(Dataset):
    """PyTorch Dataset for flow field data."""
    
    def __init__(self, input_path, mode='train'):
        self.input_path = input_path
        self.mode = mode
        self.field_types = ['rho', 'ux', 'uy', 'uz']
        
        # Load CSV metadata
        csv_path = os.path.join(input_path, f'{mode}.csv')
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)
    
    def load_binary_file(self, filepath, shape):
        """Load a binary file and reshape."""
        data = np.fromfile(filepath, dtype="<f4")
        return data.reshape(shape)
    
    def __getitem__(self, idx):
        # Set up paths
        lr_path = os.path.join(self.input_path, 'flowfields', 'LR', self.mode)
        hr_path = None if self.mode == 'test' else os.path.join(
            self.input_path, 'flowfields', 'HR', self.mode)
        
        # Load LR data
        lr_data = {}
        for field in self.field_types:
            filepath = os.path.join(lr_path, self.df[f'{field}_filename'][idx])
            lr_data[field] = self.load_binary_file(filepath, (16, 16))
        
        # Stack and normalize LR data
        X = np.stack([lr_data[field] for field in self.field_types], axis=-1)
        X_norm = normalize_batch(X, self.field_types)
        
        if self.mode == 'test':
            return torch.from_numpy(X_norm).float()
        
        # Load HR data
        hr_data = {}
        for field in self.field_types:
            filepath = os.path.join(hr_path, self.df[f'{field}_filename'][idx])
            hr_data[field] = self.load_binary_file(filepath, (128, 128))
        
        # Stack and normalize HR data
        Y = np.stack([hr_data[field] for field in self.field_types], axis=-1)
        Y_norm = normalize_batch(Y, self.field_types)
        
        return torch.from_numpy(X_norm).float(), torch.from_numpy(Y_norm).float()

def create_data_loaders(input_path, batch_size=16):
    """Create DataLoaders for training, validation, and testing."""
    train_dataset = FlowFieldDataset(input_path, 'train')
    val_dataset = FlowFieldDataset(input_path, 'val')
    test_dataset = FlowFieldDataset(input_path, 'test')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader