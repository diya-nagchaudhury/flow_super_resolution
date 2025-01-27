import os
import json
import numpy as np
import torch
import pandas as pd

from common.dataloader import load_data 
from common.visualization import visualize_results
from common.evaluation import evaluate_super_resolution
from common.config import OUTPUT_PATH
from models.nearest_interpolation import nearest_neighbour_interpolation

def save_metrics(metrics, output_path, mode):
    """Save evaluation metrics to a JSON file."""
    metrics_path = os.path.join(output_path, mode, 'metrics')
    os.makedirs(metrics_path, exist_ok=True)
    
    # Convert numpy/torch types to Python native types for JSON serialization
    processed_metrics = []
    for sample_metrics in metrics:
        sample_dict = {}
        for chan_idx, chan_metrics in enumerate(sample_metrics):
            chan_dict = {
                metric: float(value) 
                for metric, value in chan_metrics.items()
            }
            sample_dict[f'channel_{chan_idx}'] = chan_dict
        processed_metrics.append(sample_dict)
    
    # Save detailed metrics for each sample
    with open(os.path.join(metrics_path, 'detailed_metrics.json'), 'w') as f:
        json.dump(processed_metrics, f, indent=2)
    
    # Calculate and save summary statistics
    summary = {
        'average_metrics': {},
        'std_metrics': {}
    }
    
    # Calculate averages and std for each metric across all samples and channels
    all_metrics = []
    for sample in processed_metrics:
        for channel in sample.values():
            all_metrics.append(channel)
    
    metrics_names = all_metrics[0].keys()
    for metric in metrics_names:
        values = [m[metric] for m in all_metrics]
        summary['average_metrics'][metric] = float(np.mean(values))
        summary['std_metrics'][metric] = float(np.std(values))
    
    # Save summary statistics
    with open(os.path.join(metrics_path, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def process_dataset(X, Y, mode, output_dir, device):
    """Process a dataset and save results."""
    metrics_list = []
    predictions = []
    
    for idx in range(len(X)):
        # Convert to numpy for interpolation
        lr_sample = X[idx].cpu().numpy() if isinstance(X[idx], torch.Tensor) else X[idx]
        
        # Apply interpolation
        hr_sample = [
            nearest_neighbour_interpolation(img, scale_factor=8) 
            for img in lr_sample.transpose(2, 0, 1)
        ]
        predictions.append(hr_sample)
        
        # Calculate metrics if ground truth is available
        if Y is not None:
            hr_ground_truth = Y[idx].cpu().numpy() if isinstance(Y[idx], torch.Tensor) else Y[idx]
            metrics = evaluate_super_resolution(hr_sample, hr_ground_truth.transpose(2, 0, 1))
            metrics_list.append(metrics)
            
            # Print metrics for monitoring
            # for i, m in enumerate(metrics):
            #     print(f"{mode} Image {idx}, Channel {i}: "
            #           f"MSE = {m['MSE']:.4f}, "
            #           f"PSNR = {m['PSNR']:.4f} dB, "
            #           f"SSIM = {m['SSIM']:.4f}")
    
    # Save metrics if ground truth was available
    if metrics_list:
        save_metrics(metrics_list, output_dir, mode)
    
    return predictions

def main():
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all datasets
    print("Loading datasets...")
    X_train, Y_train = load_data('train')
    X_val, Y_val = load_data('val')
    X_test = load_data('test')
    
    # Move data to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    X_test = X_test.to(device)
    
    # Create base output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Process each dataset
    print("\nProcessing training set...")
    train_predictions = process_dataset(X_train, Y_train, 'train', OUTPUT_PATH, device)
    
    print("\nProcessing validation set...")
    val_predictions = process_dataset(X_val, Y_val, 'validation', OUTPUT_PATH, device)
    
    print("\nProcessing test set...")
    test_predictions = process_dataset(X_test, None, 'test', OUTPUT_PATH, device)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    visualize_results(
        X_train.cpu().numpy(), Y_train.cpu().numpy(),
        X_val.cpu().numpy(), Y_val.cpu().numpy(),
        X_test.cpu().numpy(),
        train_predictions,
        val_predictions,
        test_predictions,
        OUTPUT_PATH
    )
    
    print("\nProcessing complete! Results saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()