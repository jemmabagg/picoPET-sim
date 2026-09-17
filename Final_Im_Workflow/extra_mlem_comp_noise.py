import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from models.extramlem import extraCNN
from utils.image_ops import normalise, sum_normalise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base = "/scratch/bggjem001/pet_datasets"

#Loading test images (min-max normalised for CNN input compatibility)
test_ims_sn = np.load(f"{base}/datasets/images_test_sn.npy")
'''test_ims_mm = np.array([normalise(im) for im in test_ims_raw])  # for CNN-related ops if needed

#Sum-normalised ground truth for metric comparison
test_ims_sn = np.array([sum_normalise(im) for im in test_ims_mm])'''

scale = 182*182

n_ims = test_ims_sn.shape[0]

#Getting the MLEM iterations

it = 15
noise_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] 

global_dr = float(test_ims_sn.max() - test_ims_sn.min())

ssim_mlem, ssim_extra = [], []
psnr_mlem, psnr_extra = [], []
nmse_mlem, nmse_extra = [], []

for noise in noise_levels:

    prefix = "clean" if noise == 0 else f"noisy{noise}"
    mlem_prefix = '' if noise == 0 else f"noisy{noise}_"

    #Loading MLEM test images
    mlem_test_raw = np.load(f"{base}/datasets/{mlem_prefix}mlem{it}_test.npy")
    #mlem_test_mm = np.array([normalise(im) for im in mlem_test_raw])

    #Sum-normalised version of MLEM reconstructions for metric comparison
    mlem_test_sn = np.array([sum_normalise(im) for im in mlem_test_raw])

    mlem_test_scaled = mlem_test_sn * scale

    #Loading the trained extra-CNN model
    model = extraCNN().to(device)
    model.load_state_dict(torch.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/{prefix}_extra{it}_sn1.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        enhanced = []
        for im in mlem_test_scaled:
            x = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(x)
            enhanced.append(out.squeeze().cpu().numpy())

    enhanced = np.array(enhanced) / scale

    #NMSE
    nmse_vals = []
    for i in range(n_ims):
        nmse_vals.append(mean_squared_error(test_ims_sn[i], mlem_test_sn[i]) / np.var(test_ims_sn[i]))
    nmse_mlem.append(np.mean(nmse_vals))

    #Getting SSIM vals for normal MLEM
    ssim_vals = []
    for i in range(n_ims):
        ssim_vals.append(ssim(test_ims_sn[i], mlem_test_sn[i], data_range=global_dr))
    
    ssim_mlem.append(np.mean(ssim_vals))

    #Getting PSNR vals for normal MLEM
    psnr_vals = []
    for i in range(n_ims):
        psnr_vals.append(psnr(test_ims_sn[i], mlem_test_sn[i], data_range=global_dr))
    
    psnr_mlem.append(np.mean(psnr_vals))

    #NMSE
    nmse_vals = []
    for i in range(n_ims):
        nmse_vals.append(mean_squared_error(test_ims_sn[i], enhanced[i]) / np.var(test_ims_sn[i]))
    nmse_extra.append(np.mean(nmse_vals))

    #SSIM for extra-CNN
    ssim_vals = []
    for i in range(n_ims):
        ssim_vals.append(ssim(test_ims_sn[i], enhanced[i], data_range=global_dr))
    
    ssim_extra.append(np.mean(ssim_vals))

    #PSNR for extra-CNN
    psnr_vals = []
    for i in range(n_ims):
        psnr_vals.append(psnr(test_ims_sn[i], enhanced[i], data_range=global_dr))
    
    psnr_extra.append(np.mean(psnr_vals))
    
    #Sanity check: confirm outliers no longer dominate the comparison
    print(f"noise={noise} | "
          f"GT sum mean: {test_ims_sn.sum(axis=(1,2)).mean():.4f} | "
          f"MLEM sum mean: {mlem_test_sn.sum(axis=(1,2)).mean():.4f} | "
          f"MLEM max: {mlem_test_sn.max():.6f} | "
          f"MLEM 99th pct: {np.percentile(mlem_test_sn, 99):.6f}")

    del  mlem_test_raw, mlem_test_sn, mlem_test_scaled, enhanced, model
    torch.cuda.empty_cache()

noise_levels = np.array(noise_levels)
    

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(noise_levels*10, ssim_mlem, 'bo', label="MLEM")
axes[0].plot(noise_levels*10, ssim_extra, 'ro', label="extra-CNN")
axes[0].set_xlabel("Relative Noise Level (%)")
axes[0].set_ylabel("SSIM")
axes[0].legend()

axes[1].plot(noise_levels*10, nmse_mlem, 'bo', label="MLEM")
axes[1].plot(noise_levels*10, nmse_extra, 'ro', label="extra-CNN")
axes[1].set_xlabel("Relative Noise Level (%)")
axes[1].set_ylabel("NMSE")
axes[1].legend()

axes[2].plot(noise_levels*10, psnr_mlem, 'bo', label="MLEM")
axes[2].plot(noise_levels*10, psnr_extra, 'ro', label="extra-CNN")
axes[2].set_xlabel("Relative Noise Level (%)")
axes[2].set_ylabel("PSNR")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{base}/Plots/metrics_vs_noise_sn1.png", dpi=300, bbox_inches='tight')
plt.show()
