import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(val_loader, desc='Validation'):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            # Calculate PSNR
            mse = torch.mean((outputs - hr_imgs) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            batch_size = lr_imgs.size(0)
            total_loss += loss.item() * batch_size
            total_psnr += psnr.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    
    return avg_loss, avg_psnr

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop