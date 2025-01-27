import os
import numpy as np
import pandas as pd
import torch
from common.config import INPUT_PATH, LR_SHAPE, HR_SHAPE

class FileNotFoundOrEmptyError(Exception):
    """Custom exception for file not found or empty file cases."""
    pass

def verify_file_exists(filepath):
    """
    Verify if a file exists and is not empty.
    
    Args:
        filepath (str): Path to the file to verify
        
    Returns:
        bool: True if file exists and is not empty
        
    Raises:
        FileNotFoundOrEmptyError: If file doesn't exist or is empty
    """
    if not os.path.exists(filepath):
        raise FileNotFoundOrEmptyError(f"File not found: {filepath}")
    
    if os.path.getsize(filepath) == 0:
        raise FileNotFoundOrEmptyError(f"File is empty: {filepath}")
    
    return True

def load_binary_file(filepath, shape):
    """
    Load a binary file with error handling.
    
    Args:
        filepath (str): Path to the binary file
        shape (tuple): Expected shape of the data
        
    Returns:
        np.ndarray: Loaded and reshaped data
        
    Raises:
        ValueError: If data cannot be reshaped to expected shape
        FileNotFoundOrEmptyError: If file doesn't exist or is empty
    """
    try:
        verify_file_exists(filepath)
        data = np.fromfile(filepath, dtype="<f4")
        
        expected_size = np.prod(shape)
        if data.size != expected_size:
            raise ValueError(
                f"Data size mismatch. Expected {expected_size} elements "
                f"for shape {shape}, but got {data.size} elements"
            )
        
        return data.reshape(shape)
    
    except Exception as e:
        raise type(e)(
            f"Error loading file {filepath}: {str(e)}"
        ) from e

def load_csv_data(mode='train'):
    """
    Load CSV metadata file.
    
    Args:
        mode (str): Dataset mode ('train', 'val', or 'test')
        
    Returns:
        pd.DataFrame: Loaded CSV data
    """
    csv_path = f'{INPUT_PATH}{mode}.csv'
    try:
        verify_file_exists(csv_path)
        return pd.read_csv(csv_path)
    except Exception as e:
        raise type(e)(
            f"Error loading CSV file {csv_path}: {str(e)}"
        ) from e

def get_xy(idx, csv_file, mode='train'):
    """
    Load LR and HR data for a single sample.
    
    Args:
        idx (int): Sample index
        csv_file (pd.DataFrame): CSV metadata
        mode (str): Dataset mode ('train', 'val', or 'test')
        
    Returns:
        tuple: (LR data, HR data) for train/val, or LR data for test
    """
    try:
        # Validate paths
        LR_path = f"{INPUT_PATH}flowfields/LR/{mode}"
        HR_path = f"{INPUT_PATH}flowfields/HR/{mode}" if mode != 'test' else None
        
        if not os.path.exists(LR_path):
            raise FileNotFoundOrEmptyError(f"LR directory not found: {LR_path}")
        if HR_path and not os.path.exists(HR_path):
            raise FileNotFoundOrEmptyError(f"HR directory not found: {HR_path}")
        
        # Load LR data
        lr_files = {
            'rho': f"{LR_path}/{csv_file['rho_filename'][idx]}",
            'ux': f"{LR_path}/{csv_file['ux_filename'][idx]}",
            'uy': f"{LR_path}/{csv_file['uy_filename'][idx]}",
            'uz': f"{LR_path}/{csv_file['uz_filename'][idx]}"
        }
        
        lr_data = {
            name: load_binary_file(filepath, LR_SHAPE)
            for name, filepath in lr_files.items()
        }
        
        X = torch.stack([
            torch.from_numpy(lr_data[name]).float()
            for name in ['rho', 'ux', 'uy', 'uz']
        ], dim=2)
        
        if mode != 'test':
            # Load HR data
            hr_files = {
                'rho': f"{HR_path}/{csv_file['rho_filename'][idx]}",
                'ux': f"{HR_path}/{csv_file['ux_filename'][idx]}",
                'uy': f"{HR_path}/{csv_file['uy_filename'][idx]}",
                'uz': f"{HR_path}/{csv_file['uz_filename'][idx]}"
            }
            
            hr_data = {
                name: load_binary_file(filepath, HR_SHAPE)
                for name, filepath in hr_files.items()
            }
            
            Y = torch.stack([
                torch.from_numpy(hr_data[name]).float()
                for name in ['rho', 'ux', 'uy', 'uz']
            ], dim=2)
            
            return X, Y
        
        return X
    
    except Exception as e:
        raise type(e)(
            f"Error processing sample {idx} in mode {mode}: {str(e)}"
        ) from e

def load_data(mode='train'):
    """
    Load complete dataset.
    
    Args:
        mode (str): Dataset mode ('train', 'val', or 'test')
        
    Returns:
        tuple: (X, Y) for train/val, or X for test
    """
    try:
        print(f"Loading {mode} dataset...")
        df = load_csv_data(mode)
        print(f"Found {len(df)} samples in {mode} CSV.")
        
        data = []
        for i in range(len(df)):
            try:
                sample_data = get_xy(i, df, mode)
                data.append(sample_data)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(df)} samples...")
            except Exception as e:
                print(f"Warning: Error processing sample {i}: {str(e)}")
                continue
        
        if not data:
            raise ValueError(f"No valid samples found in {mode} dataset")
        
        if mode != 'test':
            X, Y = zip(*data)
            return torch.stack(X), torch.stack(Y)
        return torch.stack(data)
    
    except Exception as e:
        raise type(e)(
            f"Error loading {mode} dataset: {str(e)}"
        ) from e

class FlowFieldDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for flow field data."""
    
    def __init__(self, mode='train'):
        self.mode = mode
        self.df = load_csv_data(mode)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return get_xy(idx, self.df, self.mode)