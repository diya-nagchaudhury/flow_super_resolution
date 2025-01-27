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

def get_field_range(field_type):
    """Get the standardized range for each field type."""
    ranges = {
        'RHO': (0.0, 2.0),    # Density range centered around 1
        'UX': (-160.0, 160.0), # Velocity ranges all set to same scale
        'UY': (-160.0, 160.0),
        'UZ': (-160.0, 160.0)
    }
    return ranges[field_type]
            
def plot_single_channel(ax, data, title, field_type, is_error=False, colorbar=True):
    """Plot a single channel with consistent formatting and ranges."""
    if is_error:
        # For error maps, use a different colormap and fixed range
        im = ax.imshow(data, cmap='jet', vmin=0, vmax=np.percentile(data, 95))
        title = f"{title}\nMean: {np.mean(data):.2e}\nMax: {np.max(data):.2e}"
    else:
        # For regular images, use provided ranges based on field type
        vmin, vmax = get_field_range(field_type)
        im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    
    ax.set_title(title)
    ax.axis('off')
    
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    return im

def create_comparison_plot(lr_image, hr_pred, hr_ground_truth, channel_names, filename):
    """Create a comprehensive comparison plot with consistent scaling."""
    n_channels = len(channel_names)
    fig = plt.figure(figsize=(20, 4*n_channels))
    gs = GridSpec(n_channels, 4, figure=fig)
    
    for i in range(n_channels):
        # LR Image
        ax = fig.add_subplot(gs[i, 0])
        plot_single_channel(ax, lr_image[i], f'LR {channel_names[i]}', channel_names[i])
        
        # HR Predicted
        ax = fig.add_subplot(gs[i, 1])
        plot_single_channel(ax, hr_pred[i], f'Predicted {channel_names[i]}', channel_names[i])
        
        # HR Ground Truth and Error Map (if available)
        if hr_ground_truth is not None:
            # Ground Truth
            ax = fig.add_subplot(gs[i, 2])
            plot_single_channel(ax, hr_ground_truth[i], f'Ground Truth {channel_names[i]}', channel_names[i])
            
            # Error Map
            ax = fig.add_subplot(gs[i, 3])
            error_map = np.abs(hr_pred[i] - hr_ground_truth[i])
            plot_single_channel(ax, error_map, f'Error Map ({channel_names[i]})', 
                              channel_names[i], is_error=True)
    
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
    
    # Save comparison plot
    create_comparison_plot(
        lr_images, hr_pred_images,
        hr_ground_truth_images if hr_ground_truth is not None else None,
        channel_names,
        os.path.join(output_dir, mode, 'comparison', f'sample_{sample_idx}.png')
    )

def visualize_results(X_train, Y_train, X_val, Y_val, X_test, 
                     predictions_train, predictions_val, predictions_test, output_dir):
    """Visualize results for all datasets."""
    create_output_dirs(output_dir)
    
    # Visualize training samples
    for idx in range(len(X_train)):
        save_visualizations(X_train[idx], predictions_train[idx], Y_train[idx], 
                          idx, output_dir, 'train')
    
    # Visualize validation samples
    for idx in range(len(X_val)):
        save_visualizations(X_val[idx], predictions_val[idx], Y_val[idx], 
                          idx, output_dir, 'validation')
    
    # Visualize test samples (no ground truth)
    for idx in range(len(X_test)):
        save_visualizations(X_test[idx], predictions_test[idx], None, 
                          idx, output_dir, 'test')