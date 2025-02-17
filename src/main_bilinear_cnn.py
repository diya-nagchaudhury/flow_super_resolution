import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import os

input_path = '/home/diya/Projects/super_resolution/flow_super_resolution/dataset/'
#create an output directory if it doesn't exist

output_path = '/home/diya/Projects/super_resolution/flow_super_resolution/outputs/bilinear_cnn/'
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


# Convolutional upsampling model
class ConvModel(nn.Module):
    def __init__(
            self,
            channels,
            depth=4,
            width=4,
            kernel_size=15,
        ):
        super(ConvModel, self).__init__()
        self.channels = channels
        self.depth = depth
        self.width = width
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
        self.conv = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    channels * (width if d > 0 else 1),
                    channels * (width if d < depth-1 else 1), 
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    device=device,
                ),
                nn.Tanh() if d < self.depth-1 else nn.Identity(),
            ) for d in range(self.depth)
        ])

    def forward(self, u):
        batch_size_u = u.shape[0]
        channels = self.channels
        assert u.shape == (batch_size_u, channels, 16, 16)

        mean = u.mean(dim=(2, 3), keepdim=True)
        assert mean.shape == (batch_size_u, channels, 1, 1)

        std = u.std(dim=(2, 3), keepdim=True) + 1e-8
        assert std.shape == (batch_size_u, channels, 1, 1)

        # Normalize u
        u = (u - mean) / std
        assert u.shape == (batch_size_u, channels, 16, 16)

        # Upsample
        u = self.upsample(u)
        assert u.shape == (batch_size_u, channels, 128, 128)

        # Convolutional layers
        u = self.conv(u) + u
        assert u.shape == (batch_size_u, channels, 128, 128)

        # Rescale the output
        u = u * std + mean
        assert u.shape == (batch_size_u, channels, 128, 128)

        return u


# Create model
model = ConvModel(channels)
print("Model parameters:", sum([p.numel() for p in model.parameters()]))

criterion = nn.MSELoss()

# Training Loop
def train(optimizer, max_epochs, batch_size):
    losses = {"train": [], "val": []}
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epoch = 0
    start = time()
    while epoch < max_epochs:
        # Compute loss
        def eval_loss(inputs, targets):
            outputs = model(inputs)
            return criterion(outputs, targets)

        model.train()

        # Select random batch
        indices = torch.randint(0, train_input_data.shape[0], (batch_size,), device=device)
        batch_inputs = torch.index_select(train_input_data, 0, indices)
        batch_targets = torch.index_select(train_target_data, 0, indices)

        def closure():
            optimizer.zero_grad()
            loss = eval_loss(batch_inputs, batch_targets)
            loss.backward()
            return loss
        
        optimizer.step(closure)

        epoch += 1
        print(f"\rEpoch {epoch}", end="")

        if epoch % 100 == 0:
            model.eval()
            train_loss = eval_loss(train_input_data, train_target_data).item()
            val_mse = eval_loss(val_input_data, val_target_data).item()
            losses["train"].append(train_loss)
            losses["val"].append(val_mse)

            print(f"\rEpoch {epoch}:  train = {train_loss:.4e}"
                f"  val_mse = {val_mse:.4e}", end="")

            # Plot losses
            ax.clear()
            ax.plot(range(epoch//100), losses["train"], label="train")
            ax.plot(range(epoch//100), losses["val"], label="val")
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(output_path + "loss.png")

            # Save checkpoint
            #torch.save(model.state_dict(), output_path + f"checkpoint_{epoch}.pt")

    print("")
    
    # Save final model
    #torch.save(model.state_dict(), output_path + "model.pt")

    end = time()
    print(f"Took {end - start:.2f}s")


# Load model.pt
if load_model_weights:
    model.load_state_dict(
        torch.load(output_path + f"model.pt")
    )

else:
    # Train with Adam
    adam = optim.Adam(model.parameters(), lr=1e-4)
    train(adam, max_epochs=10000, batch_size=16)

    # Train with LBFGS
    lbfgs = optim.LBFGS(model.parameters(), max_iter=1, line_search_fn='strong_wolfe', history_size=5)
    train(lbfgs, max_epochs=1000, batch_size=128)


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