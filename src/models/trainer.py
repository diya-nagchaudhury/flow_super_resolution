import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.srcnn import SRCNN
from common.dataloader import create_data_loaders
from common.visualization import visualize_results
from common.normalization import denormalize_batch

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model."""
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
    
    return total_loss / total_samples, total_