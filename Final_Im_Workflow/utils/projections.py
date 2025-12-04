import numpy as np
from skimage.transform import radon, iradon
import torch

def torch_to_np(x):
    return x.detach().cpu().numpy()

def np_to_torch(x, device):
    return torch.from_numpy(x).float().to(device)

def forward_proj(image_np, theta):
    return radon(image_np, theta=theta, circle=True)

def back_proj(sino_np, theta):
    return iradon(sino_np, theta=theta, circle=True, filter_name=None)

def fp_torch(image, theta, device):
    img_np = torch_to_np(image)
    fp_np = forward_proj(img_np, theta)
    return np_to_torch(fp_np, device)

def bp_torch(sino, theta, device):
    sino_np = torch_to_np(sino)
    bp_np = back_proj(sino_np, theta)
    return np_to_torch(bp_np, device)
