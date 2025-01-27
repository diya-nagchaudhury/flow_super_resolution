import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from common.dataloader import FlowFieldDataset, load_data
from common.visualization import visualize_results
from common.evaluation import evaluate_super_resolution
from common.config import OUTPUT_PATH
from models.srcnn_old import SRCNN, init_weights
from models.trainer import train_epoch, validate, EarlyStopping

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

def process_dataset(model, X, Y, mode, output_dir, device):
    """Process a dataset and save results."""
    model.eval()
    metrics_list = []
    predictions = []
    
    with torch.no_grad():
        for idx in range(len(X)):
            # Move data to device
            lr_sample = X[idx].unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            hr_pred = model(lr_sample).squeeze(0).cpu().numpy()
            predictions.append([hr_pred[..., i] for i in range(hr_pred.shape[-1])])
            
            # Calculate metrics if ground truth is available
            if Y is not None:
                hr_ground_truth = Y[idx].cpu().numpy()
                metrics = evaluate_super_resolution(
                    predictions[-1],
                    [hr_ground_truth[..., i] for i in range(hr_ground_truth.shape[-1])]
                )
                metrics_list.append(metrics)
    
    # Save metrics if ground truth was available
    if metrics_list:
        save_metrics(metrics_list, output_dir, mode)
    
    return predictions

def main():
    # Set device and hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001
    
    # Create base output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FlowFieldDataset('train')
    val_dataset = FlowFieldDataset('val')
    X_test = load_data('test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SRCNN().to(device)
    model.apply(init_weights)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f} dB")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'best_model.pth'))
        
        # Early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'best_model.pth')))
    
    # Process all datasets
    print("\nProcessing training set...")
    X_train, Y_train = load_data('train')
    train_predictions = process_dataset(model, X_train, Y_train, 'train', OUTPUT_PATH, device)
    
    print("\nProcessing validation set...")
    X_val, Y_val = load_data('val')
    val_predictions = process_dataset(model, X_val, Y_val, 'validation', OUTPUT_PATH, device)
    
    print("\nProcessing test set...")
    test_predictions = process_dataset(model, X_test, None, 'test', OUTPUT_PATH, device)
    
    # Generate visualizations
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