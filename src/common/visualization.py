import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_output_dirs(output_path):
    """Create all necessary output directories."""
    dirs = ['LR', 'HR', 'predicted', 'comparison']
    for mode in ['train', 'validation', 'test']:
        for d in dirs:
            os.makedirs(os.path.join(output_path, mode, d), exist_ok=True)
            
def plot_single_channel(ax, data, title, vmin=None, vmax=None, is_error=False, colorbar=True):
    """Plot a single channel with consistent formatting."""
    if is_error:
        # For error maps, use a different colormap and auto-scaling
        im = ax.imshow(data, cmap='jet', vmin=0)
    else:
        # For regular images, use provided or auto vmin/vmax
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)
        im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    
    ax.set_title(title)
    ax.axis('off')
    
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    return im

def create_comparison_plot(lr_image, hr_pred, hr_ground_truth, channel_names, filename):
    """Create a comprehensive comparison plot with consistent scaling per variable."""
    n_channels = len(channel_names)
    fig = plt.figure(figsize=(20, 4*n_channels))
    gs = GridSpec(n_channels, 4, figure=fig)
    
    # Calculate value ranges for each channel separately
    value_ranges = []
    for i in range(n_channels):
        channel_data = [lr_image[i], hr_pred[i]]
        if hr_ground_truth is not None:
            channel_data.append(hr_ground_truth[i])
        
        vmin = min(np.min(data) for data in channel_data)
        vmax = max(np.max(data) for data in channel_data)
        value_ranges.append((vmin, vmax))
    
    for i in range(n_channels):
        vmin, vmax = value_ranges[i]
        
        # LR Image
        ax = fig.add_subplot(gs[i, 0])
        plot_single_channel(ax, lr_image[i], f'LR {channel_names[i]}', vmin, vmax)
        
        # HR Predicted
        ax = fig.add_subplot(gs[i, 1])
        plot_single_channel(ax, hr_pred[i], f'Predicted {channel_names[i]}', vmin, vmax)
        
        # HR Ground Truth and Error Map (if available)
        if hr_ground_truth is not None:
            # Ground Truth
            ax = fig.add_subplot(gs[i, 2])
            plot_single_channel(ax, hr_ground_truth[i], f'Ground Truth {channel_names[i]}', vmin, vmax)
            
            # Error Map
            ax = fig.add_subplot(gs[i, 3])
            error_map = np.abs(hr_pred[i] - hr_ground_truth[i])
            mean_error = np.mean(error_map)
            max_error = np.max(error_map)
            plot_single_channel(ax, error_map, 
                              f'Error Map ({channel_names[i]})\nMean: {mean_error:.2e}\nMax: {max_error:.2e}',
                              is_error=True)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def save_visualizations(lr_sample, hr_pred, hr_ground_truth, sample_idx, output_dir, mode):
    """Save all visualizations for a single sample."""
    channel_names = ['RHO', 'UX', 'UY', 'UZ']
    
    # Ensure all images are in channel-first format
    lr_images = lr_sample if lr_sample.shape[0] == len(channel_names) else lr_sample.transpose(2, 0, 1)
    hr_pred_images = hr_pred if isinstance(hr_pred, list) else hr_pred.transpose(2, 0, 1)
    if hr_ground_truth is not None:
        hr_ground_truth_images = (hr_ground_truth if hr_ground_truth.shape[0] == len(channel_names) 
                                else hr_ground_truth.transpose(2, 0, 1))
    
    # Calculate value ranges per channel for consistent plotting
    value_ranges = []
    for i in range(len(channel_names)):
        channel_data = [lr_images[i], hr_pred_images[i]]
        if hr_ground_truth is not None:
            channel_data.append(hr_ground_truth_images[i])
        vmin = min(np.min(data) for data in channel_data)
        vmax = max(np.max(data) for data in channel_data)
        value_ranges.append((vmin, vmax))
    
    # Save individual channel plots
    fig, axes = plt.subplots(1, len(channel_names), figsize=(15, 4))
    if len(channel_names) == 1:
        axes = [axes]
    
    for i, (ax, name, (vmin, vmax)) in enumerate(zip(axes, channel_names, value_ranges)):
        plot_single_channel(ax, lr_images[i], f'LR {name}', vmin, vmax)
    plt.savefig(os.path.join(output_dir, mode, 'LR', f'sample_{sample_idx}.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save predicted
    fig, axes = plt.subplots(1, len(channel_names), figsize=(15, 4))
    if len(channel_names) == 1:
        axes = [axes]
    
    for i, (ax, name, (vmin, vmax)) in enumerate(zip(axes, channel_names, value_ranges)):
        plot_single_channel(ax, hr_pred_images[i], f'Predicted {name}', vmin, vmax)
    plt.savefig(os.path.join(output_dir, mode, 'predicted', f'sample_{sample_idx}.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save ground truth if available
    if hr_ground_truth is not None:
        fig, axes = plt.subplots(1, len(channel_names), figsize=(15, 4))
        if len(channel_names) == 1:
            axes = [axes]
        
        for i, (ax, name, (vmin, vmax)) in enumerate(zip(axes, channel_names, value_ranges)):
            plot_single_channel(ax, hr_ground_truth_images[i], f'Ground Truth {name}', vmin, vmax)
        plt.savefig(os.path.join(output_dir, mode, 'HR', f'sample_{sample_idx}.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    # Save comprehensive comparison plot
    create_comparison_plot(lr_images, hr_pred_images, 
                         hr_ground_truth_images if hr_ground_truth is not None else None,
                         channel_names,
                         os.path.join(output_dir, mode, 'comparison', f'sample_{sample_idx}.png'))


def visualize_results(X_train, Y_train, X_val, Y_val, X_test, predictions_train, predictions_val, predictions_test, output_dir):
    """Visualize results for all datasets."""
    create_output_dirs(output_dir)
    
    # Visualize training samples
    for idx in range(len(X_train)):
        save_visualizations(X_train[idx], predictions_train[idx], Y_train[idx], idx, output_dir, 'train')
    
    # Visualize validation samples
    for idx in range(len(X_val)):
        save_visualizations(X_val[idx], predictions_val[idx], Y_val[idx], idx, output_dir, 'validation')
    
    # Visualize test samples (no ground truth)
    for idx in range(len(X_test)):
        save_visualizations(X_test[idx], predictions_test[idx], None, idx, output_dir, 'test')