import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN
from utils.image_ops import normalise, sum_normalise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base = "/scratch/bggjem001/pet_datasets"

#Loading test images
test_ims_sn = np.load(f"{base}/datasets/images_test_sn.npy")
#test_ims_mm = np.array([normalise(im) for im in test_ims_raw])  # for CNN-related ops if needed

#Sum-normalised ground truth for metric comparison
#test_ims_sn = np.array([sum_normalise(im) for im in test_ims_mm])

n_ims = test_ims_sn.shape[0]
scale = 182*182

#Getting the MLEM iterations

save_every = 3
iterations = list(range(save_every, 37, save_every))
noise_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
noise_levels = np.array(noise_levels)

global_dr = float(test_ims_sn.max() - test_ims_sn.min())

for noise in noise_levels:

    prefix = "clean" if noise == 0 else f"noisy{noise}"
    mlem_prefix = '' if noise == 0 else f"noisy{noise}_"

    ssim_mlem, ssim_extra = [], []
    psnr_mlem, psnr_extra = [], []
    nmse_mlem, nmse_extra = [], []

    for it in iterations:
        print(f"Evaluating iteration {it}")

        #Loading MLEM test images
        mlem_test_raw = np.load(f"{base}/datasets/{mlem_prefix}mlem{it}_test.npy")
        #mlem_test_mm = np.array([normalise(im) for im in mlem_test_raw])

        #Sum-normalised ground truth for metric comparison
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
        
        del mlem_test_raw, mlem_test_sn, enhanced, mlem_test_scaled, model
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(iterations, ssim_mlem, 'bo', label="MLEM")
    axes[0].plot(iterations, ssim_extra, 'ro', label="extra-CNN")
    axes[0].set_xlabel("MLEM Iterations")
    axes[0].set_ylabel("SSIM")
    axes[0].legend()

    axes[1].plot(iterations, nmse_mlem, 'bo', label="MLEM")
    axes[1].plot(iterations, nmse_extra, 'ro', label="extra-CNN")
    axes[1].set_xlabel("MLEM Iterations")
    axes[1].set_ylabel("NMSE")
    axes[1].legend()

    axes[2].plot(iterations, psnr_mlem, 'bo', label="MLEM")
    axes[2].plot(iterations, psnr_extra, 'ro', label="extra-CNN")
    axes[2].set_xlabel("MLEM Iterations")
    axes[2].set_ylabel("PSNR")
    axes[2].legend()

    plt.suptitle(f"Performance Metrics vs MLEM Iteration (rel noise = {noise*10}%)")

    plt.tight_layout()
    plt.savefig(f"{base}/Plots/metrics_vs_iterations_{noise*10}_sn.png", dpi=300, bbox_inches='tight')
    plt.show()

