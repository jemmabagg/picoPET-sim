import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN
from models.losses import SSIM_MSE_Loss

#Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Loading in the training, val and test data 

X_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_train.npy")
y_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem_train.npy")

X_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_val.npy")
y_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem_val.npy")

train_dataset = MLEMDataset(y_train, X_train)
val_dataset = MLEMDataset(y_val, X_val)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

#Training 
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
epochs = 30
results = {}

for alpha in alphas:

    print(f"Starting training for alpha = {alpha}")

    model = extraCNN().to(device)
    loss_function = SSIM_MSE_Loss(alpha=alpha)
    optim = torch.optim.Adam(model.parameters(), lr=0.003)

    train_loss, val_loss, train_mse, val_mse, train_ssim, val_ssim = [], [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss, running_train_mse, running_train_ssim = 0, 0, 0

        for mlem_im, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):

            mlem_im = mlem_im.to(device).float()
            target = target.to(device).float()

            output = model(mlem_im)
            output = output.unsqueeze(0)
            target = target.unsqueeze(0)
            loss = loss_function(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

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

            for val_mlem_im, val_target in tqdm(val_loader, desc="Validation", leave=False):

                val_mlem_im = val_mlem_im.to(device).float()
                val_target = val_target.to(device).float()

                val_output = model(val_mlem_im)
                val_output = val_output.unsqueeze(0)
                val_target = val_target.unsqueeze(0)
                v_loss = loss_function(val_output, val_target)
                running_val_loss += v_loss.item()
                v_mse, v_ssim_ = loss_function.components(val_output, val_target)
                running_val_mse += v_mse
                running_val_ssim += v_ssim_

            val_loss.append(running_val_loss / len(val_loader))
            val_mse.append(running_val_mse / len(val_loader))
            val_ssim.append(running_val_ssim / len(val_loader))

    #Saving the model
    torch.save(model.state_dict(), f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extramlem_alpha_{alpha}.pth")
    
    #Saving the loss functions
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_train_loss_alpha_{alpha}.npy", train_loss)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_val_loss_alpha_{alpha}.npy", val_loss)

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_train_mse_alpha_{alpha}.npy", train_mse)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_val_mse_alpha_{alpha}.npy", val_mse)

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_train_ssim_alpha_{alpha}.npy", train_ssim)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_val_ssim_alpha_{alpha}.npy", val_ssim)

