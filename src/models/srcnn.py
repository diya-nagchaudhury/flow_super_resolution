import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class SRCNN(nn.Module):
    def __init__(self, upscale_factor=8):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        
        # Define unified scales for velocity components
        # Using -160 to 160 as the unified range for all velocity components
        self.scales = {
            'rho': {'center': 1.0, 'scale': 5.0},  # Keep density scaling as is
            'ux': {'center': 0.0, 'scale': 160.0},  # Center at 0, scale to ±160
            'uy': {'center': 0.0, 'scale': 160.0},  # Use same scale as UX
            'uz': {'center': 0.0, 'scale': 160.0}   # Use same scale as UX
        }
        
        # Density pathway
        self.conv1_rho = nn.Conv2d(1, 32, kernel_size=9, padding=4)
        self.relu1_rho = nn.ReLU(inplace=True)
        self.conv2_rho = nn.Conv2d(32, 16, kernel_size=1, padding=0)
        self.relu2_rho = nn.ReLU(inplace=True)
        self.conv3_rho = nn.Conv2d(16, 1, kernel_size=5, padding=2)
        
        # Unified velocity pathway (since all components now use same scale)
        self.conv1_vel = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1_vel = nn.ReLU(inplace=True)
        self.conv2_vel = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2_vel = nn.ReLU(inplace=True)
        self.conv3_vel = nn.Conv2d(32, 3, kernel_size=5, padding=2)
    
    def bicubic_upscale(self, x):
        """Upscale input using bicubic interpolation."""
        batch_size, channels, height, width = x.shape
        upscaled = []
        
        for b in range(batch_size):
            channels_list = []
            for c in range(channels):
                img = x[b, c].cpu().numpy()
                upscaled_img = cv2.resize(
                    img, 
                    (width * self.upscale_factor, height * self.upscale_factor), 
                    interpolation=cv2.INTER_CUBIC
                )
                channels_list.append(torch.from_numpy(upscaled_img))
            upscaled.append(torch.stack(channels_list))
        
        return torch.stack(upscaled).to(x.device)
    
    def normalize_component(self, x, component_name):
        """Normalize a component based on its scale."""
        scale_info = self.scales[component_name]
        return (x - scale_info['center']) / scale_info['scale']
    
    def denormalize_component(self, x, component_name):
        """Denormalize a component back to its original scale."""
        scale_info = self.scales[component_name]
        return x * scale_info['scale'] + scale_info['center']
    
    def forward(self, x):
        # Input shape: (batch_size, height, width, channels)
        # Convert to: (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Split into density and velocity components
        rho = x[:, 0:1, :, :]
        vel = x[:, 1:, :, :]  # All velocity components
        
        # Process density
        rho_up = self.bicubic_upscale(rho)
        rho_norm = self.normalize_component(rho_up, 'rho')
        rho_out = self.relu1_rho(self.conv1_rho(rho_norm))
        rho_out = self.relu2_rho(self.conv2_rho(rho_out))
        rho_out = self.conv3_rho(rho_out)
        rho_out = self.denormalize_component(rho_out, 'rho')
        
        # Process velocity components together
        vel_up = self.bicubic_upscale(vel)
        # Normalize all velocity components to same scale
        vel_norm = torch.stack([
            self.normalize_component(vel_up[:, i:i+1], comp)
            for i, comp in enumerate(['ux', 'uy', 'uz'])
        ], dim=1).squeeze(2)
        
        vel_out = self.relu1_vel(self.conv1_vel(vel_norm))
        vel_out = self.relu2_vel(self.conv2_vel(vel_out))
        vel_out = self.conv3_vel(vel_out)
        
        # Denormalize velocity components
        vel_out = torch.stack([
            self.denormalize_component(vel_out[:, i:i+1], comp)
            for i, comp in enumerate(['ux', 'uy', 'uz'])
        ], dim=1).squeeze(2)
        
        # Combine outputs
        out = torch.cat([rho_out, vel_out], dim=1)
        
        # Convert back to: (batch_size, height, width, channels)
        return out.permute(0, 2, 3, 1)

def init_weights(m):
    """Initialize model weights using Kaiming initialization."""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)