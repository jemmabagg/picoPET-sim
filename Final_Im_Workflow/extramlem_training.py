import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.extramlem import extraCNN
from models.mlem_dataset import MLEMDataset
from utils.image_ops import normalise
import gc

#Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

images_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_train.npy")
images_val   = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_val.npy")

images_train = np.array([normalise(im) for im in images_train])
images_val   = np.array([normalise(im) for im in images_val])

save_every = 3

for it in range(1, 40):

    # Skip iterations we do not want to save
    if it % save_every != 0:
        continue

    print(f"Running MLEM iteration {it}")

    #Loading in the training, val and test data 
    mlem_images_train = np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem{it}_train.npy")
    mlem_images_val = np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem{it}_val.npy")

    #Normalise the reconstructed images
    mlem_images_train = np.array([normalise(im) for im in mlem_images_train])
    mlem_images_val   = np.array([normalise(im) for im in mlem_images_val])

    train_dataset = MLEMDataset(mlem_images_train, images_train)
    val_dataset = MLEMDataset(mlem_images_val, images_val)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)

    #Training the model
    model = extraCNN().to(device)
    loss_function = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss, val_loss = [], []
    epochs = 30

    for epoch in range(epochs):

        model.train()
        running_train_loss = 0

        for mlem_im, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):

            mlem_im = mlem_im.to(device).float()
            target = target.to(device).float()

            output = model(mlem_im)
            loss = loss_function(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_train_loss += loss.item()

        train_loss.append(running_train_loss / len(train_loader))

        #Validation 
        model.eval()
        running_val_loss = 0
 
        with torch.no_grad():

            for val_mlem_im, val_target in tqdm(val_loader, desc="Validation", leave=False):

                val_mlem_im = val_mlem_im.to(device).float()
                val_target = val_target.to(device).float()

                val_output = model(val_mlem_im)
                v_loss = loss_function(val_output, val_target)
                running_val_loss += v_loss.item()

        val_loss.append(running_val_loss / len(val_loader))
        print(f"Epoch {epoch+1} | Train Loss: {train_loss[-1]:.6f} | Val Loss: {val_loss[-1]:.6f}")

    #Saving the model
    torch.save(model.state_dict(), f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra{it}.pth")

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra{it}_train_loss.npy", train_loss)
    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra{it}_val_loss.npy", val_loss)

    print(f"Model {it} Trained")

    del mlem_images_train, mlem_images_val, train_dataset, val_dataset, model
    torch.cuda.empty_cache()
    gc.collect()

print("All models trained")




    

