import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from common.dataloader import FlowFieldDataset
from common.dataloader import load_data
from common.visualization_srcnn import visualize_results
from common.evaluation import evaluate_super_resolution
from common.config import OUTPUT_PATH
from models.srcnn import SRCNN, init_weights
from common.normalization import normalize_batch, denormalize_batch

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
    
    # Calculate summary statistics
    summary = {
        'average_metrics': {},
        'std_metrics': {}
    }
    
    metrics_names = list(processed_metrics[0]['channel_0'].keys())
    for metric in metrics_names:
        values = []
        for sample in processed_metrics:
            for channel in sample.values():
                values.append(channel[metric])
        summary['average_metrics'][metric] = float(np.mean(values))
        summary['std_metrics'][metric] = float(np.std(values))
    
    with open(os.path.join(metrics_path, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    field_types = ['rho', 'ux', 'uy', 'uz']
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0
    field_types = ['rho', 'ux', 'uy', 'uz']
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(val_loader, desc='Validation'):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            # Calculate PSNR on denormalized data
            outputs_cpu = outputs.cpu().numpy()
            hr_imgs_cpu = hr_imgs.cpu().numpy()
            
            outputs_denorm = denormalize_batch(outputs_cpu, field_types)
            hr_imgs_denorm = denormalize_batch(hr_imgs_cpu, field_types)
            
            mse = np.mean((outputs_denorm - hr_imgs_denorm) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            
            batch_size = lr_imgs.size(0)
            total_loss += loss.item() * batch_size
            total_psnr += psnr * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples, total_psnr / total_samples

def process_dataset(model, loader, mode, output_dir, device):
    """Process a dataset and save results."""
    model.eval()
    metrics_list = []
    predictions = []
    field_types = ['rho', 'ux', 'uy', 'uz']
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f'Processing {mode}')):
            if mode != 'test':
                lr_imgs, hr_imgs = batch
                hr_imgs = hr_imgs.to(device)
            else:
                lr_imgs = batch
            
            lr_imgs = lr_imgs.to(device)
            outputs = model(lr_imgs)
            
            # Denormalize predictions
            outputs_denorm = denormalize_batch(
                outputs.cpu().numpy(), field_types
            )
            
            # Store predictions
            for b in range(outputs_denorm.shape[0]):
                pred = [outputs_denorm[b, :, :, i] for i in range(4)]
                predictions.append(pred)
                
                if mode != 'test':
                    # Denormalize ground truth
                    hr_denorm = denormalize_batch(
                        hr_imgs[b].cpu().numpy(), field_types
                    )
                    hr_channels = [hr_denorm[:, :, i] for i in range(4)]
                    
                    # Calculate metrics
                    metrics = evaluate_super_resolution(pred, hr_channels)
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
    
    # Create datasets and loaders
    train_dataset = FlowFieldDataset('train')
    val_dataset = FlowFieldDataset('val')
    test_dataset = FlowFieldDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize model
    model = SRCNN().to(device)
    model.apply(init_weights)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
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
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'best_model.pth')))
    
    # Load data for visualization
    print("\nLoading data for visualization...")
    X_train, Y_train = load_data('train')
    X_val, Y_val = load_data('val')
    X_test = load_data('test')
    
    # Process datasets
    print("\nProcessing training set...")
    train_predictions = process_dataset(model, train_loader, 'train', OUTPUT_PATH, device)
    
    print("\nProcessing validation set...")
    val_predictions = process_dataset(model, val_loader, 'validation', OUTPUT_PATH, device)
    
    print("\nProcessing test set...")
    test_predictions = process_dataset(model, test_loader, 'test', OUTPUT_PATH, device)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(
        X_train.numpy(), Y_train.numpy(),
        X_val.numpy(), Y_val.numpy(),
        X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test,
        train_predictions,
        val_predictions,
        test_predictions,
        OUTPUT_PATH
    )
    
    print("\nProcessing complete! Results saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()