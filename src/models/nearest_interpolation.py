import cv2
import numpy as np

def nearest_neighbour_interpolation_opencv(image, scale_factor):
    height, width = image.shape
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def nearest_neighbour_interpolation(image, scale_factor):
    """
    Perform 2D nearest neighbor interpolation to upscale an image.
    Parameters:
        image (numpy.ndarray): Input image as a 2D numpy array.
        scale_factor (int): Factor by which to increase the image size.
    Returns:
        numpy.ndarray: Upscaled image.
    """
    height, width = image.shape
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    # Create the output image
    upscaled_image = np.zeros((new_height, new_width), dtype=image.dtype)
    
    # Calculate the corresponding coordinates in the original image
    x_ratio = width / new_width
    y_ratio = height / new_height
    
    for i in range(new_height):
        for j in range(new_width):
            # Find the nearest neighbor coordinates in the original image
            src_x = min(width - 1, int(j * x_ratio))
            src_y = min(height - 1, int(i * y_ratio))
            
            # Copy the pixel value from the nearest neighbor
            upscaled_image[i, j] = image[src_y, src_x]
    
    return upscaled_image