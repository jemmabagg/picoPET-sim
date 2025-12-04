import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.transform import radon, iradon

#Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

#Loading in the training, val and test data 

X_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_train.npy")
y_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_train.npy")

X_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_val.npy")
y_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_val.npy")

#Class to get the datasets into a format pytorch can use

class MLEMDataset(Dataset):
    def __init__(self, sinograms, images):
        self.sinograms = sinograms
        self.images = images

    def __len__(self):
        return(len(self.sinograms))

    #Enables us to get one item at a time
    def __getitem__(self, idx):
        sino = torch.from_numpy(self.sinograms[idx]).float()
        img = torch.from_numpy(self.images[idx]).float()
        return sino, img

train_dataset = MLEMDataset(y_train, X_train)
val_dataset = MLEMDataset(y_val, X_val)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

#Functions
def np_to_torch(x):
    return torch.from_numpy(x).float().to(device)

def torch_to_np(x):
    return x.detach().cpu().numpy()

def forward_proj(image_np):
    return radon(image_np, theta=theta, circle=True)

def back_proj(sino_np):
    return iradon(sino_np, theta=theta, circle=True, filter_name=None)

#Torch forward and backprojections
def fp_torch(image):
    return np_to_torch(forward_proj(torch_to_np(image)))

def bp_torch(sino):
    return np_to_torch(back_proj(torch_to_np(sino)))

#Base CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=(3,3)), nn.PReLU(),
            nn.Conv2d(8, 8, 5, padding=(2,2)), nn.PReLU(),
            nn.Conv2d(8, 8, 3, padding=(1,1)), nn.PReLU(),
            nn.Conv2d(8, 8, 5, padding=(2,2)), nn.PReLU(),
            nn.Conv2d(8, 1, 7, padding=(3,3))
        )

    def forward(self, x):
        return x + self.net(x)

cnn = CNN().to(device)

#intra-MLEM model
class intraCNN(nn.Module):
    def __init__(self, cnn, num_its):
        super().__init__()
        self.cnn = cnn
        self.num_its = num_its

    def forward(self, sino):
        sino = sino.to(device)
        sens = bp_torch(torch.ones_like(sino))
        recon = torch.ones_like(sino)
        iters = self.num_its

        for _ in range(iters):
            fp = fp_torch(recon)
            ratio = sino / (fp + 1e-9)
            correction = bp_torch(ratio) / (sens + 1e-9)
            recon = recon * correction

            #CNN Enhancement
            recon = recon.unsqueeze(0).unsqueeze(0) #[1, 1, H, W]
            recon = recon + self.cnn(recon)
            recon = recon.squeeze(0).squeeze(0) #[H, W]
            recon = torch.abs(recon)

        return recon

#instantiate model
num_its = 10
model = intraCNN(cnn, num_its).to(device)

#Training
loss_function = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.003)

train_loss, val_loss = [], []
epochs = 20

for epoch in range(epochs):
    model.train()
    running_train_loss = 0

    for sino, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):

        sino = sino.squeeze(0).to(device)
        target = target.squeeze(0).to(device)

        output = model(sino)
        loss = loss_function(output, target)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        running_train_loss += loss.item()
    
    train_loss.append(running_train_loss / len(train_loader))

    #Validation
    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for val_sino, val_target in tqdm(val_loader, desc="Validation", leave=False):
            val_sino = val_sino.squeeze(0).to(device)
            val_target = val_target.squeeze(0).to(device)
            val_output = model(val_sino)
            v_loss = loss_function(val_output, val_target)
            running_val_loss += v_loss.item()

    val_loss.append(running_val_loss / len(val_loader))
    print(f"Epoch {epoch+1} | Train Loss: {train_loss[-1]:.6f} | Val Loss: {val_loss[-1]:.6f}")

#Saving the model
torch.save(model.state_dict(), f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intracnn.pth")

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_train_loss.npy", train_loss)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/intra_val_loss.npy", val_loss)

print("Model Trained")