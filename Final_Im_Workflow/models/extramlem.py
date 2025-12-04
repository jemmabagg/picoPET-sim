import torch
import torch.nn as nn
 
#Defining the extra-MLEM architecture
class extraCNN(nn.Module):
    def __init__(self):
        super(extraCNN, self).__init__()
        self.CNN_denoise = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=3), nn.PReLU(),
            nn.Conv2d(8, 8, 5, padding=2), nn.PReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.PReLU(),
            nn.Conv2d(8, 8, 5, padding=2), nn.PReLU(),
            nn.Conv2d(8, 1, 7, padding=3)
        )

    def forward(self, x):
        return x + self.CNN_denoise(x)