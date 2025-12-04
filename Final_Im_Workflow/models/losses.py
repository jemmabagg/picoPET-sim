import torch
import torch.nn as nn
from pytorch_msssim import ssim as ssim_torch

#Combined MSE and SSIM loss function
class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = (1 - ssim_torch(pred, target, data_range=1.0, size_average=True))

        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
    
    def components(self, pred, target):
        #Return individual MSE and SSIM components
        mse_val = self.mse(pred, target).item()
        ssim_val = 1 - ssim_torch(pred, target, data_range=1.0, size_average=True).item()
        return mse_val, ssim_val