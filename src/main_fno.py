import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import os

input_path = '/home/diya/Projects/super_resolution/flow_super_resolution/dataset/'
#create an output directory if it doesn't exist

output_path = '/home/diya/Projects/super_resolution/flow_super_resolution/outputs/fno/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

load_model_weights = False

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load data
train_df = pd.read_csv(input_path + 'train.csv')

print("Training samples:", len(train_df))

val_df = pd.read_csv(input_path + 'val.csv')

variables = ['rho', 'ux', 'uy', 'uz']
channels = len(variables)
size = len(train_df)

# Load sample
def load_sample(idx, res, train="train"):
    assert train in ["train", "val"]
    assert res in ["HR", "LR"]

    n = 128 if res == "HR" else 16
    df = train_df if train == "train" else val_df
    data_path = input_path + f"flowfields/{res}/{train}"

    sample = {"idx": idx, "res": res.lower()}
    for v in variables:
        filename = df[f'{v}_filename'][idx]
        #print(sample[v])
        sample[v] = np.fromfile(data_path + "/" + filename, dtype="<f4").reshape(n, n)
    return sample

# Convert sample to tensor
def sample_to_tensor(sample):
    return torch.stack([
        torch.from_numpy(sample[v]).to(device)
        for v in variables]
    )

# Load a whole dataset
def load_dataset(train, res, size=None):
    assert train in ["train", "val"]
    df = train_df if train == "train" else val_df
    if size is None:
        size = len(df)
    samples = []
    for i in range(size):
        sample = sample_to_tensor(load_sample((i)%len(df), res, train))
        samples.append(sample)
    return torch.stack(samples)

# Plotting
def plot(sample, postfix=""):
    fig, axs = plt.subplots(1, channels, figsize=(5*channels, 5))
    try:
        axs.shape
    except:
        axs = [axs]
    for i, v in enumerate(variables):
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = axs[i].imshow(sample[v], cmap='jet')
        fig.colorbar(im, cax=cax, orientation='vertical')
        axs[i].set_title(v)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    filename = output_path + f"{sample['res']}{postfix}.png"
    fig.savefig(filename, dpi=300)


train_input_data = load_dataset("train", "LR", size)
train_target_data = load_dataset("train", "HR", size)
print("Training samples:", size)

val_input_data = load_dataset("val", "LR")
val_target_data = load_dataset("val", "HR")

assert train_input_data.shape == (size, channels, 16, 16)
assert train_target_data.shape == (size, channels, 128, 128)



    
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, max = N/2 + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Compute Fourier coefficients up to the Nyquist frequency
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply only the lower modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", 
                x_ft[:, :, :self.modes1, :self.modes2], 
                self.weights1)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x

class FNO2d(nn.Module):
    def __init__(
            self, 
            channels,
            modes1=8,        # Reduced number of Fourier modes
            modes2=8,
            width=32,        # Reduced width
            depth=4,         # Number of FNO layers
            activation='gelu' # Activation function
        ):
        super(FNO2d, self).__init__()
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.activation = getattr(F, activation)
        
        # Input layer
        self.fc0 = nn.Linear(channels, width)
        
        # FNO layers
        self.conv_layers = nn.ModuleList([])
        self.w_layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.conv_layers.append(SpectralConv2d(width, width, modes1, modes2))
            self.w_layers.append(nn.Conv2d(width, width, 1))
        
        # Output layer with gradual dimension reduction
        self.fc1 = nn.Linear(width, max(width, channels))
        self.fc2 = nn.Linear(max(width, channels), channels)
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, u):
        batch_size = u.shape[0]
        
        # Normalize input
        mean = u.mean(dim=(2, 3), keepdim=True)
        std = u.std(dim=(2, 3), keepdim=True) + 1e-8
        u = (u - mean) / std

        # Upsample to target resolution
        u = self.upsample(u)
        
        # Lift channel dimension
        u = self.fc0(u.permute(0, 2, 3, 1))
        u = u.permute(0, 3, 1, 2)

        # FNO layers with gradient checkpointing for memory efficiency
        for i in range(self.depth):
            u1 = self.conv_layers[i](u)
            u2 = self.w_layers[i](u)
            u = u1 + u2
            if i < self.depth - 1:  # Apply activation except for last layer
                u = self.activation(u)
                # Free memory
                del u1, u2
                torch.cuda.empty_cache()

        # Project back to physical space
        u = self.fc1(u.permute(0, 2, 3, 1))
        u = self.activation(u)
        u = self.fc2(u)
        u = u.permute(0, 3, 1, 2)

        # Rescale output
        u = u * std + mean
        
        return u

# Create model with reduced memory footprint
model = FNO2d(
    channels=channels,
    modes1=4,          # Reduced number of modes
    modes2=4,          # Reduced number of modes
    width=16,          # Reduced width
    depth=4,           # Same depth
    activation='gelu'
).to(device)

print("Model parameters:", sum([p.numel() for p in model.parameters()]))

criterion = nn.MSELoss()

# # Training Loop
# def train(optimizer, max_epochs, batch_size):
#     losses = {"train": [], "val": []}
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     epoch = 0
#     start = time()
#     while epoch < max_epochs:
#         # Compute loss
#         def eval_loss(inputs, targets):
#             outputs = model(inputs)
#             return criterion(outputs, targets)

#         model.train()

#         # Select random batch
#         indices = torch.randint(0, train_input_data.shape[0], (batch_size,), device=device)
#         batch_inputs = torch.index_select(train_input_data, 0, indices)
#         batch_targets = torch.index_select(train_target_data, 0, indices)

#         def closure():
#             optimizer.zero_grad()
#             loss = eval_loss(batch_inputs, batch_targets)
#             loss.backward()
#             return loss
        
#         optimizer.step(closure)

#         epoch += 1
#         print(f"\rEpoch {epoch}", end="")

#         if epoch % 100 == 0:
#             model.eval()
#             train_loss = eval_loss(train_input_data, train_target_data).item()
#             val_mse = eval_loss(val_input_data, val_target_data).item()
#             losses["train"].append(train_loss)
#             losses["val"].append(val_mse)

#             print(f"\rEpoch {epoch}:  train = {train_loss:.4e}"
#                 f"  val_mse = {val_mse:.4e}", end="")

#             # Plot losses
#             ax.clear()
#             ax.plot(range(epoch//100), losses["train"], label="train")
#             ax.plot(range(epoch//100), losses["val"], label="val")
#             ax.set_xlabel("Epoch")
#             ax.set_yscale("log")
#             ax.legend()
#             fig.savefig(output_path + "loss.png")

#             # Save checkpoint
#             #torch.save(model.state_dict(), output_path + f"checkpoint_{epoch}.pt")

#     print("")
    
#     # Save final model
#     #torch.save(model.state_dict(), output_path + "model.pt")

#     end = time()
#     print(f"Took {end - start:.2f}s")


# # Load model.pt
# if load_model_weights:
#     model.load_state_dict(
#         torch.load(output_path + f"model.pt")
#     )

# else:
#     # Train with Adam
#     adam = optim.Adam(model.parameters(), lr=1e-4)
#     train(adam, max_epochs=10000, batch_size=16)

#     # Train with LBFGS
#     lbfgs = optim.LBFGS(model.parameters(), max_iter=1, line_search_fn='strong_wolfe', history_size=5)
#     train(lbfgs, max_epochs=1000, batch_size=128)

# Modify training loop to include gradient accumulation for memory efficiency
def train(optimizer, max_epochs, batch_size, accumulation_steps=4):
    losses = {"train": [], "val": []}
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epoch = 0
    start = time()
    
    while epoch < max_epochs:
        model.train()
        optimizer.zero_grad()
        
        # Split batch into smaller chunks
        for i in range(0, batch_size, accumulation_steps):
            # Select random mini-batch
            indices = torch.randint(0, train_input_data.shape[0], (accumulation_steps,), device=device)
            mini_batch_inputs = torch.index_select(train_input_data, 0, indices)
            mini_batch_targets = torch.index_select(train_target_data, 0, indices)
            
            # Forward pass
            outputs = model(mini_batch_inputs)
            loss = criterion(outputs, mini_batch_targets) / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Free memory
            del outputs, loss
            torch.cuda.empty_cache()
        
        optimizer.step()
        
        epoch += 1
        print(f"\rEpoch {epoch}", end="")
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(train_input_data), train_target_data).item()
                val_mse = criterion(model(val_input_data), val_target_data).item()
                losses["train"].append(train_loss)
                losses["val"].append(val_mse)
                
                print(f"\rEpoch {epoch}:  train = {train_loss:.4e} val_mse = {val_mse:.4e}", end="")
                
                # Plot losses
                ax.clear()
                ax.plot(range(epoch//100), losses["train"], label="train")
                ax.plot(range(epoch//100), losses["val"], label="val")
                ax.set_xlabel("Epoch")
                ax.set_yscale("log")
                ax.legend()
                fig.savefig(output_path + "loss.png")
    
    print("")
    print(f"Took {time() - start:.2f}s")


# Plot the model output
test_idx = 1
plot(load_sample(test_idx, "HR"))
plot(load_sample(test_idx, "LR"))

test_input_data = sample_to_tensor(load_sample(test_idx, "LR")).unsqueeze(0)
test_output_data = model(test_input_data)[0].cpu().detach().numpy()
test_sample = {"idx": test_idx, "res": "hr"}
for i, v in enumerate(variables):
    test_sample[v] = test_output_data[i, :, :]
plot(test_sample, postfix="_pred")

# Compute MSE training error
train_loss = criterion(model(train_input_data), train_target_data)
print(f"Training MSE   = {train_loss.item():.4e}")

# Compute MSE validation error
val_loss = criterion(model(val_input_data), val_target_data)
print(f"Validation MSE = {val_loss.item():.4e}")


# Define means and std to weigh density and velocity predictions (NECESSARY for submission!)
my_mean = [0.24, 28.0, 28.0, 28.0]
my_std = [0.068, 48.0, 48.0, 48.0]
my_mean = np.array(my_mean)
my_std = np.array(my_std)

test_df = pd.read_csv(input_path + '/test.csv')

# Gets test set input
def getTestX(idx):
    csv_file = test_df.reset_index().to_dict(orient='list')
    LR_path = input_path + "flowfields/LR/test" 
    id = csv_file['id'][idx]

    rho_i = np.fromfile(LR_path + "/" + csv_file['rho_filename'][idx], dtype="<f4").reshape(16, 16)
    ux_i = np.fromfile(LR_path + "/" + csv_file['ux_filename'][idx], dtype="<f4").reshape(16, 16)
    uy_i = np.fromfile(LR_path + "/" + csv_file['uy_filename'][idx], dtype="<f4").reshape(16, 16)
    uz_i = np.fromfile(LR_path + "/" + csv_file['uz_filename'][idx], dtype="<f4").reshape(16, 16)
    rho_i = torch.from_numpy(rho_i)
    ux_i = torch.from_numpy(ux_i)
    uy_i = torch.from_numpy(uy_i)
    uz_i = torch.from_numpy(uz_i)

    X = torch.stack([rho_i, ux_i, uy_i, uz_i]).unsqueeze(0).to(device)
    assert X.shape == (1, 4, 16, 16)
    return id, X

# Predicts with input
def predict(idx, model):
    id, X = getTestX(idx)
    assert X.shape == (1, 4, 16, 16)
    y_pred = model(X)
    assert y_pred.shape == (1, 4, 128, 128)
    y_pred = y_pred.transpose(1, 2).transpose(2, 3)
    assert y_pred.shape == (1, 128, 128, 4)
    return id, y_pred

# Generates submission with model predictions
def generate_submission(model):
    y_preds = {}
    ids = []
    for idx in range(len(test_df)):
        id, y_pred = predict(idx, model)
        #this normalizes density and velocity to be in the same range
        tmp = (y_pred.cpu().detach().numpy() - my_mean)/my_std 
        y_preds[id]= tmp.flatten(order='C').astype(np.float32)
        ids.append(id)
    df = pd.DataFrame.from_dict(y_preds, orient='index')
    df['id'] = ids
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df = df.reset_index(drop=True)
    return df

df = generate_submission(model)
df.to_csv(output_path + 'submission.csv', index=False)