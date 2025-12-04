import torch.nn as nn
import torch
from utils.projections import fp_torch, bp_torch

#Base CNN architecture (for intra)
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

class intraCNN(nn.Module):
    def __init__(self, cnn, num_its, device, theta):
        super().__init__()
        self.cnn = cnn
        self.num_its = num_its
        self.device = device
        self.theta = theta

    def forward(self, sino):
        # Move sino to device
        sino = sino.to(self.device)

        # Sensitivity image
        sens = bp_torch(torch.ones_like(sino), self.theta, self.device)

        # Initial MLEM estimate
        recon = torch.ones_like(sino)

        for _ in range(self.num_its):

            # Forward projection
            fp = fp_torch(recon, self.theta, self.device)

            # Ratio
            ratio = sino / (fp + 1e-9)

            # Backproject ratio
            correction = bp_torch(ratio, self.theta, self.device) / (sens + 1e-9)

            # Update reconstruction
            recon = recon * correction

            # CNN enhancement
            recon = recon.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            recon = recon + self.cnn(recon)
            recon = recon.squeeze(0).squeeze(0)      # [H,W]

            recon = torch.abs(recon)

        return recon
