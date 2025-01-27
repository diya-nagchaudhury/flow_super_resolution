import cv2
import numpy as np

def bicubic_interpolation_opencv(image, scale_factor):
    height, width = image.shape
    new_height, new_width = height * scale_factor, width * scale_factor
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def cubic_interpolation(p, x):
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))

def bicubic_interpolation(image, scale_factor):
    height, width = image.shape
    new_height, new_width = height * scale_factor, width * scale_factor
    
    padded_image = np.pad(image, 2, mode='edge')
    upscaled_image = np.zeros((new_height, new_width))
    
    for i in range(new_height):
        for j in range(new_width):
            x, y = i / scale_factor, j / scale_factor
            x_int, y_int = int(x), int(y)
            x_frac, y_frac = x - x_int, y - y_int
            
            neighborhood = padded_image[x_int:x_int+4, y_int:y_int+4]
            
            col0 = cubic_interpolation(neighborhood[:, 0], x_frac)
            col1 = cubic_interpolation(neighborhood[:, 1], x_frac)
            col2 = cubic_interpolation(neighborhood[:, 2], x_frac)
            col3 = cubic_interpolation(neighborhood[:, 3], x_frac)
            
            upscaled_image[i, j] = cubic_interpolation([col0, col1, col2, col3], y_frac)
    
    return upscaled_image