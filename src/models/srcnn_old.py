import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class SRCNN(nn.Module):
    def __init__(self, upscale_factor=8):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        
        # Feature extraction layer (conv1)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping layer (conv2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction layer (conv3)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=5, padding=2)
        
    def bicubic_upscale(self, x):
        """Upscale input using bicubic interpolation."""
        # Convert to numpy for OpenCV processing
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
    
    def forward(self, x):
        # Input shape: (batch_size, height, width, channels)
        # Convert to: (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Bicubic upscaling first
        x_upscaled = self.bicubic_upscale(x)
        
        # Apply SRCNN on upscaled image
        out = self.relu1(self.conv1(x_upscaled))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        
        # Convert back to: (batch_size, height, width, channels)
        return out.permute(0, 2, 3, 1)

def init_weights(m):
    """Initialize model weights using Kaiming initialization."""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)