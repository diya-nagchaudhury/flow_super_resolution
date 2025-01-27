import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2, data_range=None):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1, img2: Input images
        data_range: Dynamic range of the images (max - min)
    """
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def gaussian_kernel(size=11, sigma=1.5):
    """Generate a 2D Gaussian kernel."""
    x = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, x)
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    return kernel / kernel.sum()

def calculate_ssim(img1, img2, data_range=None, win_size=11, sigma=1.5):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1, img2: Input images
        data_range: Dynamic range of the images (max - min)
        win_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
    """
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    # Constants to prevent division by zero
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Generate Gaussian kernel
    kernel = gaussian_kernel(win_size, sigma)
    
    # Compute means
    mu1 = gaussian_filter(img1, sigma=sigma, mode='reflect')
    mu2 = gaussian_filter(img2, sigma=sigma, mode='reflect')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma, mode='reflect') - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma, mode='reflect') - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma, mode='reflect') - mu1_mu2
    
    # Compute SSIM
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    
    return np.mean(ssim_map)

def calculate_nrmse(img1, img2, normalization='min-max'):
    """
    Calculate Normalized Root Mean Square Error.
    
    Args:
        img1, img2: Input images
        normalization: Normalization method ('min-max' or 'mean')
    """
    mse = calculate_mse(img1, img2)
    rmse = np.sqrt(mse)
    
    if normalization == 'min-max':
        return rmse / (img2.max() - img2.min())
    elif normalization == 'mean':
        return rmse / np.mean(img2)
    else:
        raise ValueError("Normalization must be 'min-max' or 'mean'")

def calculate_uqi(img1, img2, win_size=8):
    """
    Calculate Universal Quality Index (UQI).
    
    Args:
        img1, img2: Input images
        win_size: Size of the sliding window
    """
    N = win_size ** 2
    sum1 = np.zeros_like(img1, dtype=np.float64)
    sum2 = np.zeros_like(img2, dtype=np.float64)
    sum12 = np.zeros_like(img1, dtype=np.float64)
    sqsum1 = np.zeros_like(img1, dtype=np.float64)
    sqsum2 = np.zeros_like(img2, dtype=np.float64)
    
    # Sliding window implementation
    sum1 = uniform_filter(img1, size=win_size)
    sum2 = uniform_filter(img2, size=win_size)
    sum12 = uniform_filter(img1 * img2, size=win_size)
    sqsum1 = uniform_filter(img1 ** 2, size=win_size)
    sqsum2 = uniform_filter(img2 ** 2, size=win_size)
    
    # Calculate means and cross-correlation
    mean1 = sum1
    mean2 = sum2
    cross = sum12 - mean1 * mean2
    var1 = sqsum1 - mean1 ** 2
    var2 = sqsum2 - mean2 ** 2
    
    # Calculate UQI
    numerator = 4 * cross * mean1 * mean2
    denominator = (var1 + var2) * (mean1**2 + mean2**2)
    
    # Handle division by zero
    mask = denominator != 0
    q_map = np.zeros_like(numerator)
    q_map[mask] = numerator[mask] / denominator[mask]
    
    return np.mean(q_map)

def evaluate_super_resolution(hr_images, hr_ground_truth):
    """
    Evaluate super-resolution results using multiple metrics.
    
    Args:
        hr_images: List of predicted high-resolution images
        hr_ground_truth: List of ground truth high-resolution images
    """
    data_range = np.max(hr_ground_truth) - np.min(hr_ground_truth)
    metrics = []
    
    for pred, true in zip(hr_images, hr_ground_truth):
        mse = calculate_mse(pred, true)
        psnr = calculate_psnr(pred, true, data_range=data_range)
        ssim = calculate_ssim(pred, true, data_range=data_range)
        nrmse = calculate_nrmse(pred, true)
        uqi = calculate_uqi(pred, true)
        
        metrics.append({
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim,
            'NRMSE': nrmse,
            'UQI': uqi
        })
    
    return metrics