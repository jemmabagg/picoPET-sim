import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.transform import radon, iradon
from pytorch_msssim import ssim as ssim_torch
from pytorch_msssim import ms_ssim
from models.mlem_dataset import MLEMDataset
from models.intramlem import CNN, intraCNN
from models.losses import SSIM_MSE_Loss

#Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

#Loading in the training, val and test data 

X_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_train.npy")
y_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_train.npy")

X_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_val.npy")
y_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_val.npy")

train_dataset = MLEMDataset(y_train, X_train)
val_dataset = MLEMDataset(y_val, X_val)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

#instantiate model
num_its = 10
epochs = 20

for alpha in alphas:
    print(f"Starting training for alpha = {alpha}")

    #Training
    cnn = CNN().to(device)
    model = intraCNN(cnn, num_its, device, theta).to(device)
    loss_function = SSIM_MSE_Loss(alpha=alpha)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.003)

    train_loss, val_loss, train_mse, val_mse, train_ssim, val_ssim = [], [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss, running_train_mse, running_train_ssim = 0, 0, 0

        for sino, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):

            sino = sino.squeeze(0).to(device)
            target = target.squeeze(0).to(device)

            output = model(sino)
            loss = loss_function(output, target)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            running_train_loss += loss.item()
            mse, ssim_ = loss_function.components(output, target)
            running_train_mse += mse
            running_train_ssim += ssim_
    
        train_loss.append(running_train_loss / len(train_loader))
        train_mse.append(running_train_mse / len(train_loader))
        train_ssim.append(running_train_ssim / len(train_loader))

        #Validation
        model.eval()
        running_val_loss, running_val_mse, running_val_ssim = 0, 0, 0
        with torch.no_grad():
            for val_sino, val_target in tqdm(val_loader, desc="Validation", leave=False):
                val_sino = val_sino.squeeze(0).to(device)
                val_target = val_target.squeeze(0).to(device)
                val_output = model(val_sino)
                v_loss = loss_function(val_output, val_target)
                running_val_loss += v_loss.item()

                v_mse, v_ssim = loss_function.components(val_output, val_target)
                running_val_mse += v_mse
                running_val_ssim += v_ssim

        val_loss.append(running_val_loss / len(val_loader))
        val_mse.append(running_val_mse / len(val_loader))
        val_ssim.append(running_val_ssim / len(val_loader))
        print(f"Epoch {epoch+1} | Train Loss: {train_loss[-1]:.6f} | Val Loss: {val_loss[-1]:.6f}")

    #Saving the model
    torch.save(model.state_dict(), f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intramlem_alpha_{alpha}.pth")
    
    #Saving the loss functions
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intra_train_loss_alpha_{alpha}.npy", train_loss)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intra_val_loss_alpha_{alpha}.npy", val_loss)

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intra_train_mse_alpha_{alpha}.npy", train_mse)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intra_val_mse_alpha_{alpha}.npy", val_mse)

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intra_train_ssim_alpha_{alpha}.npy", train_ssim)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_combined/intra_val_ssim_alpha_{alpha}.npy", val_ssim)


