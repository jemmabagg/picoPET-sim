import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN
from utils.image_ops import normalise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base = "/scratch/bggjem001/picoPET-sim/Final_Im_Workflow"

#Loading test images
test_ims = np.load(f"{base}/datasets/images_test.npy")
test_ims = np.array([normalise(im) for im in test_ims])

n_ims = test_ims.shape[0]

#Getting the MLEM iterations

save_every = 3
iterations = list(range(save_every, 40, save_every))

ssim_mlem, ssim_extra = [], []

for it in iterations:
    print(f"Evaluating iteration {it}")

    #Loading MLEM test images
    mlem_test = np.load(f"{base}/datasets/mlem{it}_test.npy")
    mlem_test = np.array([normalise(im) for im in mlem_test])

    #Getting SSIM vals for normal MLEM
    ssim_vals = []
    for i in range(n_ims):
        ssim_vals.append(ssim(test_ims[i], mlem_test[i], data_range=1.0))
    
    ssim_mlem.append(np.mean(ssim_vals))

    #Loading the trained extra-CNN model
    model = extraCNN().to(device)
    model.load_state_dict(torch.load(f"{base}/model_weights/extra{it}.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        enhanced = []
        for im in mlem_test:
            x = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(x)
            enhanced.append(out.squeeze().cpu().numpy())

    #SSIM for extra-CNN
    ssim_vals = []
    for i in range(n_ims):
        ssim_vals.append(ssim(test_ims[i], enhanced[i], data_range=1.0))
    
    ssim_extra.append(np.mean(ssim_vals))
    
    del mlem_test, enhanced, model
    torch.cuda.empty_cache()

plt.plot(iterations, ssim_mlem, 'bo', label="MLEM")
plt.plot(iterations, ssim_extra, 'ro', label="extra-CNN")
plt.xlabel("MLEM Iterations")
plt.ylabel("Average SSIM")
plt.legend()
plt.savefig(f"{base}/Plots/ssim_vs_iterations.png")
plt.show()
