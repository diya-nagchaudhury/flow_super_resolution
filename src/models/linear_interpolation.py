import cv2
import numpy as np

def linear_interpolation_opencv(image, scale_factor):
    height, width = image.shape
    new_height, new_width = height * scale_factor, width * scale_factor
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def linear_interpolation(image, scale_factor):
    """
    Perform 2D linear interpolation to upscale an image.
    
    Parameters:
    image (numpy.ndarray): Input image as a 2D numpy array.
    scale_factor (int): Factor by which to increase the image size.
    
    Returns:
    numpy.ndarray: Upscaled image.
    """
    height, width = image.shape
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # Create coordinate grids for the original and new image sizes
    x = np.linspace(0, width - 1, new_width)
    y = np.linspace(0, height - 1, new_height)
    x_coords, y_coords = np.meshgrid(x, y)
    
    # Perform linear interpolation
    upscaled_image = cv2.remap(image, x_coords.astype(np.float32), y_coords.astype(np.float32), 
                               interpolation=cv2.INTER_LINEAR)
    
    return upscaled_image