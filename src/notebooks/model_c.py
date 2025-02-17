import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Normalize
from typing import List, Dict, Tuple, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1. Interpolation Model
class BicubicBaseline(nn.Module):
    def __init__(self, scale_factor=8):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

# 2. SRCNN Model
class SRCNN(nn.Module):
    def __init__(self, in_channels=4, scale_factor=8):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # Increase network capacity and adjust architecture for flow fields
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
        
        # Initialize weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial bicubic upsampling
        bicubic = F.interpolate(
            x, 
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )
        
        # Feature extraction
        x = self.conv1(bicubic)
        
        # Non-linear mapping
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reconstruction
        x = self.conv4(x)
        
        # Residual connection with bicubic upsampling
        return x + bicubic

class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downsampling block with maxpool followed by double convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upsampling block with either transpose conv or bilinear upsampling"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle cases where input dimensions don't match perfectly
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=64, bilinear=False, scale_factor=8):
        """
        UNet implementation for flow field super-resolution
        
        Args:
            in_channels (int): Number of input channels (default: 4 for density + 3 velocity components)
            out_channels (int): Number of output channels (same as input for super-resolution)
            features (int): Number of base features (channels) in first layer
            bilinear (bool): Whether to use bilinear upsampling instead of transpose convolutions
            scale_factor (int): The super-resolution scale factor (default: 8)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.scale_factor = scale_factor
        
        # Initial double convolution
        self.inc = DoubleConv(in_channels, features)
        
        # Downsampling path
        self.down1 = DownBlock(features, features * 2)
        self.down2 = DownBlock(features * 2, features * 4)
        self.down3 = DownBlock(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features * 8, features * 16 // factor)
        
        # Upsampling path
        self.up1 = UpBlock(features * 16, features * 8 // factor, bilinear)
        self.up2 = UpBlock(features * 8, features * 4 // factor, bilinear)
        self.up3 = UpBlock(features * 4, features * 2 // factor, bilinear)
        self.up4 = UpBlock(features * 2, features, bilinear)
        
        # Final convolution
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Modified super-resolution upsampling path
        # Calculate the log2 of scale factor to determine number of upsampling steps
        n_upsamples = int(np.log2(scale_factor))
        sr_layers = []
        current_scale = 1
        
        for _ in range(n_upsamples):
            sr_layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            current_scale *= 2
            
        # Add final convolution
        sr_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
        self.sr_up = nn.Sequential(*sr_layers)

    def forward(self, x):
        # Store input for residual connection
        input_tensor = x
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Upsampling path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final convolutions
        x = self.outc(x)
        
        # Super-resolution upsampling
        x = self.sr_up(x)
        
        # Add residual connection with bicubic upsampled input
        x_bicubic = F.interpolate(
            input_tensor, 
            size=x.shape[2:],  # Use the size of x instead of scale_factor
            mode='bicubic', 
            align_corners=False
        )
        
        return x + x_bicubic

# 3. ResNet Model
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class ResNetSR(nn.Module):
    def __init__(self, in_channels=4, num_filters=64, num_res_blocks=16, scale_factor=8):
        super(ResNetSR, self).__init__()
        self.scale_factor = scale_factor
        
        self.conv_input = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        res_blocks = [ResBlock(num_filters) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv_mid = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        
        upsampling = []
        for _ in range(int(np.log2(scale_factor))):
            upsampling.extend([
                nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        self.upsampling = nn.Sequential(*upsampling)
        self.conv_output = nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        bicubic = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        out = self.conv_input(x)
        residual = out
        out = self.res_blocks(out)
        out = self.conv_mid(out)
        out += residual
        out = self.upsampling(out)
        out = self.conv_output(out)
        return out + bicubic

# 4. FNO Model
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1  
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        # Complex weights for each mode
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        # Calculate modes to use based on input size
        modes1 = min(self.modes1, size1//2)
        modes2 = min(self.modes2, size2//2)
        
        # First multiplication for lower frequencies
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2],
            torch.view_as_complex(self.weights1[..., :modes1, :modes2, :])
        )
            
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(size1, size2), norm='ortho')
        return x

class FNO(nn.Module):
    def __init__(self, in_channels=4, modes=12, width=32, scale_factor=8):
        super(FNO, self).__init__()
        self.scale_factor = scale_factor
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        # Initial convolution layers
        self.conv0 = nn.Conv2d(in_channels, self.width, kernel_size=1)
        
        # Fourier layers
        self.conv_list = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(4)
        ])
        
        # Regular convolution layers after Fourier layers
        self.w_list = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=1)
            for _ in range(4)
        ])
        
        # Upsampling network
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False),
            nn.Conv2d(self.width, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # Initial convolution
        x = self.conv0(x)
        
        # Fourier layers
        for i, (conv, w) in enumerate(zip(self.conv_list, self.w_list)):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            if i < len(self.conv_list) - 1:  # Don't apply activation to last layer
                x = F.gelu(x)
        
        # Upsampling
        x = self.upsample(x)
        return x

# 5. Bicubic-FNO Model (Combines Bicubic upsampling with FNO refinement)
class BicubicFNO(nn.Module):
    def __init__(self, in_channels=4, modes=12, width=32, scale_factor=8):
        super(BicubicFNO, self).__init__()
        self.scale_factor = scale_factor
        self.fno = FNO(in_channels, modes, width, scale_factor=1)  # FNO works at high resolution
        
    def forward(self, x):
        # Bicubic upsampling first
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        # FNO refinement
        x_refined = self.fno(x_up)
        return x_refined + x_up
    
# 6. Blended-FNO Model (Combines Bicubic upsampling with FNO refinement and blending)

class BlendedFNO(nn.Module):
    def __init__(self, in_channels=4, modes=12, width=32, scale_factor=8):
        super(BlendedFNO, self).__init__()
        self.scale_factor = scale_factor
        self.fno = FNO(in_channels, modes, width, scale_factor=1)  # FNO works at high resolution
        
        # Learnable blending parameter
        # self.blend_matrix = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.blend_matrix = torch.nn.Parameter(torch.zeros(1, in_channels, 128, 128), requires_grad=True)
        
    def forward(self, x):
        # Bicubic upsampling first
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        # FNO refinement
        x_refined = self.fno(x_up)
        # Blend bicubic upsampled image with FNO output
        # print("self.blend_matrix:\n", self.blend_matrix, self.blend_matrix.min(), self.blend_matrix.max())
        return (1 - self.blend_matrix) * x_up + (self.blend_matrix) * x_refined
    
def custom_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> float:
    """
    Calculate SSIM with proper handling of edge cases.
    
    Args:
        img1: First image
        img2: Second image
        win_size: Size of the sliding window
        
    Returns:
        float: SSIM value
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions')
    
    # Use a minimum window size of 3
    win_size = max(3, win_size)
    
    # If image is smaller than window, use image size as window
    win_size = min(win_size, min(img1.shape))
    if win_size % 2 == 0:
        win_size -= 1
        
    # Create window
    window = np.ones((win_size, win_size)) / (win_size ** 2)
    
    # Calculate means
    mu1 = np.average(img1)
    mu2 = np.average(img2)
    
    # Calculate variances and covariance
    sigma1_sq = np.average((img1 - mu1) ** 2)
    sigma2_sq = np.average((img2 - mu2) ** 2)
    sigma12 = np.average((img1 - mu1) * (img2 - mu2))
    
    # Calculate SSIM
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    if den == 0:
        return 0.0
        
    return num / den

def calculate_metrics(output: Union[torch.Tensor, np.ndarray], 
                        target: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Calculate MSE, SSIM, and PSNR metrics with proper tensor/array handling.
    
    Args:
        output: Model output tensor/array
        target: Target tensor/array
        
    Returns:
        Dict containing MSE, SSIM, and PSNR values
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(output, torch.Tensor):
        output_np = output.detach().cpu().numpy().squeeze()
    else:
        output_np = output.squeeze()
        
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().squeeze()
    else:
        target_np = target.squeeze()
    
    # Handle edge cases
    if output_np.size == 0 or target_np.size == 0:
        return {
            'mse': float('inf'),
            'ssim': 0.0,
            'psnr': 0.0
        }
    
    # Calculate MSE
    mse = np.mean((output_np - target_np) ** 2)
    
    # Get minimum dimension for window size
    min_dim = min(output_np.shape)
    win_size = min(7, min_dim - 1 if min_dim % 2 == 0 else min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    
    # Normalize for PSNR calculation
    output_norm = (output_np - output_np.min()) / (output_np.max() - output_np.min() + 1e-8)
    target_norm = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)
    
    # Calculate PSNR
    try:
        psnr_val = psnr(target_norm, output_norm)
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        psnr_val = 0.0
    
    # Calculate SSIM
    try:
        ssim_val = custom_ssim(output_np, target_np,
                       win_size=win_size)
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        ssim_val = 0.0
    
    return {
        'mse': float(mse),
        'ssim': float(ssim_val),
        'psnr': float(psnr_val)
    }

def calculate_channel_metrics(output: Union[torch.Tensor, np.ndarray], 
                            target: Union[torch.Tensor, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each channel separately.
    
    Args:
        output: Model output tensor/array [C, H, W]
        target: Target tensor/array [C, H, W]
        
    Returns:
        Dict containing metrics for each channel
    """
    # Convert tensors to numpy if needed
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    metrics = {}
    channels = ['density', 'velocity_x', 'velocity_y', 'velocity_z']
    
    for i, channel in enumerate(channels):
        metrics[channel] = calculate_metrics(output[i], target[i])
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics.values()]),
        'ssim': np.mean([m['ssim'] for m in metrics.values()]),
        'psnr': np.mean([m['psnr'] for m in metrics.values()])
    }
    
    metrics['average'] = avg_metrics
    
    return metrics

def calculate_channel_metrics(output: torch.Tensor, target: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each channel separately.
    
    Args:
        output: Model output tensor [C, H, W]
        target: Target tensor [C, H, W]
        
    Returns:
        Dict containing metrics for each channel
    """
    metrics = {}
    channels = ['density', 'velocity_x', 'velocity_y', 'velocity_z']
    
    for i, channel in enumerate(channels):
        output_channel = output[i].cpu().detach().numpy()
        target_channel = target[i].cpu().detach().numpy()
        
        metrics[channel] = calculate_metrics(output_channel, target_channel)
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics.values()]),
        'ssim': np.mean([m['ssim'] for m in metrics.values()]),
        'psnr': np.mean([m['psnr'] for m in metrics.values()])
    }
    
    metrics['average'] = avg_metrics
    
    return metrics


class FlowFieldDataset:
    def __init__(self, input_path, mode="test"):
        self.mode = mode
        self.csv_file = pd.read_csv(input_path + f"{mode}.csv")
        self.LR_path = input_path + f"flowfields/LR/{mode}"
        self.HR_path = input_path + f"flowfields/HR/{mode}"
        
        self.mean = np.array([0.24, 28.0, 28.0, 28.0])
        self.std = np.array([0.068, 48.0, 48.0, 48.0])
        
    def transform(self, x):
        return Compose([ToTensor(), Normalize(self.mean, self.std, inplace=True)])(x)
        
    def __len__(self):
        return len(self.csv_file)
        
    def __getitem__(self, idx):
        # Load LR data
        rho_i = np.fromfile(f"{self.LR_path}/{self.csv_file['rho_filename'][idx]}", dtype="<f4").reshape(16, 16)
        ux_i = np.fromfile(f"{self.LR_path}/{self.csv_file['ux_filename'][idx]}", dtype="<f4").reshape(16, 16)
        uy_i = np.fromfile(f"{self.LR_path}/{self.csv_file['uy_filename'][idx]}", dtype="<f4").reshape(16, 16)
        uz_i = np.fromfile(f"{self.LR_path}/{self.csv_file['uz_filename'][idx]}", dtype="<f4").reshape(16, 16)
        
        # Load HR data
        rho_o = np.fromfile(f"{self.HR_path}/{self.csv_file['rho_filename'][idx]}", dtype="<f4").reshape(128, 128)
        ux_o = np.fromfile(f"{self.HR_path}/{self.csv_file['ux_filename'][idx]}", dtype="<f4").reshape(128, 128)
        uy_o = np.fromfile(f"{self.HR_path}/{self.csv_file['uy_filename'][idx]}", dtype="<f4").reshape(128, 128)
        uz_o = np.fromfile(f"{self.HR_path}/{self.csv_file['uz_filename'][idx]}", dtype="<f4").reshape(128, 128)
        
        X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)
        Y = np.stack([rho_o, ux_o, uy_o, uz_o], axis=2)
        
        return self.transform(X), self.transform(Y)

def load_model_weights(model, checkpoint_path):
    """Load saved model weights."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"No weights found at {checkpoint_path}, using initialized weights")
    return model

def create_output_dirs(base_path):
    """Create necessary output directories."""
    dirs = ['plots', 'metrics', 'predictions']
    paths = {}
    for dir_name in dirs:
        path = os.path.join(base_path, dir_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        paths[dir_name] = path
    return paths

def train_model(model, train_loader, val_loader, 
              criterion, optimizer, scheduler,
              num_epochs, device, checkpoint_dir,
              model_name, save_best=True):
    """
    Training loop with validation and model checkpointing.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save model checkpoints
        model_name: Name of the model for saving checkpoints
        save_best: Whether to save best model based on validation loss
    """
    
    # Initialize best validation loss for model saving
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'mse': [], 'ssim': [], 'psnr': []
        }
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_pbar):
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate metrics
                metrics = calculate_channel_metrics(outputs[0], targets[0])
                for metric in val_metrics:
                    val_metrics[metric].append(metrics['average'][metric])
                
                # Update running loss
                val_loss += loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{val_loss/(batch_idx+1):.4f}'
                })
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in val_metrics.items()
        }
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print('Validation Metrics:')
        for metric, value in avg_metrics.items():
            print(f'{metric.upper()}: {value:.4f}')
        
        # Save checkpoint if it's the best model so far
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': avg_metrics
            }
            torch.save(checkpoint, 
                      os.path.join(checkpoint_dir, f'{model_name}_best.pth'))
            print(f'Saved best model checkpoint with validation loss: {val_loss:.4f}')
        
        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
    
    return train_losses, val_losses

def train_all_models(train_loader, val_loader, device, base_path, config):
    """
    Train all models with specified configurations.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        base_path: Base path for saving outputs
        config: Dictionary containing training configurations
    """
    
    # Common training settings
    num_epochs = config['num_epochs']
    checkpoint_dir = os.path.join(base_path, "checkpoints")
    
    # Initialize models
    models = {
        'SRCNN': SRCNN().to(device),
        'UNet': UNet().to(device),
        'ResNet': ResNetSR().to(device),
        'FNO': FNO().to(device),
        'Bicubic-FNO': BicubicFNO().to(device),
        'Blended-FNO': BlendedFNO().to(device)
    }
    
    # Training configurations for each model
    training_configs = {
        'SRCNN': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'UNet': {  # Add configuration for UNet
        'lr': 1e-4,
        'weight_decay': 1e-5
        },
        'ResNet': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'FNO': {
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'Bicubic-FNO': {
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'Blended-FNO': {
            'lr': 1e-3,
            'weight_decay': 1e-4
        }
    }
    
    # Train each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_configs[model_name]['lr'],
            weight_decay=training_configs[model_name]['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
            verbose=True
        )
        
        # Set up loss function
        criterion = nn.MSELoss()
        
        # Train the model
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name.lower(),
            save_best=True
        )
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training Curves')
        plt.legend()
        plt.savefig(os.path.join(base_path, f'{model_name.lower()}_training_curves.png'))
        plt.close()
        
def plot_comparison(models: Dict[str, nn.Module], test_batch: Tuple[torch.Tensor, torch.Tensor],
                   save_path: str = 'model_comparison.png'):
    """
    Create comparison plot with a cleaner layout showing:
    - LR input and HR target
    - All model predictions
    - Error maps for Bicubic, Bicubic-FNO and Blended-FNO
    """
    inputs, targets = test_batch
    channels = ['Density', 'X-Velocity', 'Y-Velocity', 'Z-Velocity']
    error_map_models = ['Bicubic', 'Bicubic-FNO', 'Blended-FNO']
    
    # Calculate layout
    n_base_cols = 2  # Input + Target
    n_model_cols = len(models)  # Model predictions
    n_error_cols = len(error_map_models)  # Error maps
    total_cols = n_base_cols + n_model_cols + n_error_cols
    
    # Create figure with the correct number of columns
    fig, axs = plt.subplots(len(channels), total_cols, figsize=(25, 16))
    plt.suptitle('Flow Field Super Resolution Model Comparison', fontsize=16)
    
    # Dictionary to store metrics
    all_metrics = {}
    
    # Plot LR inputs and HR targets (first two columns)
    for i, channel in enumerate(channels):
        # Plot LR input
        im = axs[i, 0].imshow(inputs[0, i].cpu().detach().numpy(), cmap='viridis')
        axs[i, 0].set_title(f'LR Input\n{channel}' if i == 0 else channel)
        plt.colorbar(im, ax=axs[i, 0])
        
        # Plot HR target
        im = axs[i, 1].imshow(targets[0, i].cpu().detach().numpy(), cmap='viridis')
        axs[i, 1].set_title('HR Target' if i == 0 else '')
        plt.colorbar(im, ax=axs[i, 1])
    
    # Plot model predictions
    model_outputs = {}  # Store outputs for error map calculation
    for idx, (name, model) in enumerate(models.items()):
        col = idx + n_base_cols  # Start after input and target
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            model_outputs[name] = outputs  # Store for error maps
            
            # Calculate metrics for each channel
            channel_metrics = []
            for c in range(4):
                output_channel = outputs[0, c:c+1]
                target_channel = targets[0, c:c+1]
                metrics = calculate_metrics(output_channel, target_channel)
                channel_metrics.append(metrics)
                
                # Plot prediction
                im = axs[c, col].imshow(outputs[0, c].cpu().detach().numpy(), cmap='viridis')
                title = f'{name}\nPSNR: {metrics["psnr"]:.2f}' if c == 0 else ''
                axs[c, col].set_title(title)
                plt.colorbar(im, ax=axs[c, col])
            
            # Average metrics
            avg_metrics = {
                'mse': np.mean([m['mse'] for m in channel_metrics]),
                'ssim': np.mean([m['ssim'] for m in channel_metrics]),
                'psnr': np.mean([m['psnr'] for m in channel_metrics])
            }
            all_metrics[name] = avg_metrics
    
    # Plot error maps for selected models
    error_start_col = n_base_cols + n_model_cols
    for idx, model_name in enumerate(error_map_models):
        if model_name in model_outputs:
            outputs = model_outputs[model_name]
            col = error_start_col + idx
            
            for c in range(4):
                error = outputs[0, c].cpu().detach().numpy() - targets[0, c].cpu().detach().numpy()
                max_err = np.abs(error).max()
                im_error = axs[c, col].imshow(
                    error, 
                    cmap='RdBu',
                    norm=plt.Normalize(vmin=-max_err, vmax=max_err)
                )
                title = f'{model_name}\nError Map' if c == 0 else ''
                axs[c, col].set_title(title)
                plt.colorbar(im_error, ax=axs[c, col])
    
    # Remove axis ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print metrics summary
    print("\nMetrics Summary:")
    print("-" * 80)
    print(f"{'Model':<15} {'MSE':>10} {'SSIM':>10} {'PSNR':>10}")
    print("-" * 80)
    for name, metrics in all_metrics.items():
        print(f"{name:<15} {metrics['mse']:>10.6f} {metrics['ssim']:>10.4f} {metrics['psnr']:>10.2f}")
    print("-" * 80)
    
    return all_metrics


# Example usage in main:
if __name__ == "__main__":
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths and directories
    base_path = "/home/diya/Projects/super_resolution/flow_super_resolution/"
    input_path = os.path.join(base_path, "dataset/")
    output_path = os.path.join(base_path, "outputs/model_comparison/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    paths = create_output_dirs(output_path)
    
    # Training configuration
    config = {
        'num_epochs': 100,
        'batch_size': 32,
        'num_workers': 4
    }
    
    # Set up data loaders
    train_dataset = FlowFieldDataset(input_path=input_path, mode="train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_dataset = FlowFieldDataset(input_path=input_path, mode="val")
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Train all models
    train_all_models(train_loader, val_loader, device, base_path, config)
    
    # Initialize models
    models = {
        'Bicubic': BicubicBaseline().to(device),
        'SRCNN': SRCNN().to(device),
        'UNet': UNet().to(device),
        'ResNet': ResNetSR().to(device),
        'FNO': FNO().to(device),
        'Bicubic-FNO': BicubicFNO().to(device),
        'Blended-FNO': BlendedFNO().to(device)
    }
    
    # Load pretrained weights
    checkpoint_dir = os.path.join(base_path, "checkpoints")
    for model_name, model in models.items():
        if model_name != 'Bicubic':  # Bicubic doesn't need weights
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name.lower()}_best.pth")
            models[model_name] = load_model_weights(model, checkpoint_path)
        model.eval()
    
    # Process test samples and generate comparisons
    all_metrics = []
    print("Generating comparisons for test samples...")
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Generate comparison plot
        plot_path = os.path.join(paths['plots'], f'comparison_sample_{batch_idx}_custom.png')
        
        # Basic usage
        metrics = plot_comparison(models, (inputs, targets), save_path=plot_path)
        
        

        # With custom settings
        # metrics = plot_comparison(
        #     models, 
        #     (inputs, targets),
        #     channel_names=['Density', 'Vx', 'Vy', 'Vz'],
        #     save_path=plot_path,
        #     plot_error_maps=True,
        #     vmin=-1.0,
        #     vmax=1.0
        # )
        
        #metrics = plot_comparison(models, (inputs, targets), save_path=plot_path)
        
        # Store metrics
        metrics['batch_idx'] = batch_idx
        all_metrics.append(metrics)
        
        # Save first 5 samples only to save space
        if batch_idx >= 10:
            break
    
    # Calculate and save average metrics
    avg_metrics = {
        model_name: {
            metric: np.mean([m[model_name][metric] for m in all_metrics])
            for metric in ['mse', 'ssim', 'psnr']
        }
        for model_name in models.keys()
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(avg_metrics).round(4)
    metrics_df.to_csv(os.path.join(paths['metrics'], 'average_metrics.csv'))
    
    # Print final summary
    print("\nFinal Average Metrics:")
    print("-" * 80)
    print(f"{'Model':<15} {'MSE':>10} {'SSIM':>10} {'PSNR':>10}")
    print("-" * 80)
    for model_name, metrics in avg_metrics.items():
        print(f"{model_name:<15} {metrics['mse']:>10.6f} {metrics['ssim']:>10.4f} {metrics['psnr']:>10.2f}")
    print("-" * 80)
    #print(output_path)
    print(f"\nOutputs saved to: {output_path}")
    print("Done!")