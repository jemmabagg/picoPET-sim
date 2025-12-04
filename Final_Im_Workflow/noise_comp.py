import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.noise_fcn import scal_func, add_poisson, gaussian
from skimage.transform import radon, iradon
from models.extramlem import extraCNN
from utils.mlem import mlem_reco
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

theta = np.linspace(0., 180., 180, endpoint=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Loading in the data 

images_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_images_test.npy")

#Getting the average counts 

av_counts = []
sinos_clean = []

for im in images_test:

    sino_clean = radon(im, theta, circle=True)
    sinos_clean.append(sino_clean)

    counts_bin = sino_clean.ravel()
    bin_heights, bin_edges = np.histogram(counts_bin, bins=50)
    bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

    # 1) Clean/weight the data (avoid zeros/NaNs and use Poisson weights)
    mask = np.isfinite(bin_heights) & np.isfinite(bin_centres) & (bin_heights > 0)
    x = bin_centres[mask]
    y = bin_heights[mask]
    sigma_w = np.sqrt(y)            # Poisson stdev
    sigma_w[sigma_w == 0] = 1.0     # safety

    # 2) Data-driven initial guesses
    A0 = y.max()
    # weighted mean & std as starting points
    mu0 = np.average(x, weights=y)
    sigma0 = np.sqrt(np.average((x - mu0)**2, weights=y))
    # fallback if sigma0 is tiny/NaN
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = (x.max() - x.min())/6.0

    p0 = [A0, mu0, sigma0]

    # 3) Constrain parameters to sensible ranges and allow more iterations
    lower = [0, x.min(), 1e-9]      # A>=0, mu in data range, sigma>0
    upper = [np.inf, x.max(), np.inf]

    popt, pcov = curve_fit(
        gaussian, x, y,
        p0=p0,
        sigma=sigma_w,
        absolute_sigma=True,   # interpret sigma as true stdevs
        bounds=(lower, upper), # enables robust 'trf' solver internally
        maxfev=20000           # more function evaluations
        )

    av_counts.append(popt[0])

## EVALUATING HOW EACH ALGORITHM PERFORMS ON ONE IMAGE ##

idx = 50
true_image_np = images_test[idx]
sino = sinos_clean[idx]
av_count = av_counts[idx]

#noise_levels = [None, 1, 3, 5, 7, 10]
noise_levels = [0.05]

mse_extra, ssim_extra = [], []
mse_mlem10, ssim_mlem10 = [], []

## EXTRA ##

extra_train_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/noisy_test_extra_train_loss.npy")
extra_val_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/noisy_test_extra_val_loss.npy")

plt.plot(extra_train_loss, label="Train Loss")
plt.plot(extra_val_loss, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_noisy_test_loss_plot.png", dpi=300, bbox_inches='tight')
plt.show()

#Loading in the model
model = extraCNN().to(device)
model.load_state_dict(torch.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/_noisy_test_extracnn.pth"))
model.eval()

with torch.no_grad():
    for noise in noise_levels:

        if noise is None:
            noisy_sino = sino
        else:
            scale = scal_func(noise, av_count)
            noisy_sino = add_poisson(sino, scale)
        
        recon_mlem = mlem_reco(noisy_sino, theta, 10)
        inp = torch.from_numpy(recon_mlem).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        out = model(inp)                                    # [1,1,H,W]
        mlem_cnn_extra = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

        mse_extra_val = mean_squared_error(true_image_np, mlem_cnn_extra)
        ssim_val_denoise = ssim(true_image_np, mlem_cnn_extra, data_range=true_image_np.max() - true_image_np.min())
        mse_extra.append(mse_extra_val)
        ssim_extra.append(ssim_val_denoise)

        mse_mlem10_val = mean_squared_error(true_image_np, recon_mlem)
        ssim_val_mlem10= ssim(true_image_np, recon_mlem, data_range=true_image_np.max() - true_image_np.min())
        mse_mlem10.append(mse_mlem10_val)
        ssim_mlem10.append(ssim_val_mlem10)
        #mlem_cnn_extra_ims.append(mlem_cnn_extra)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # <- use fig and axes

# MSE plot
#axes[0].plot(noise_levels, mse_scores_cnn, 'bo', label=f'Intra-CNN ({num_its} Iterations)')
#axes[0].plot(noise_levels, mse_scores_mlem20, 'go', label=f'MLEM (20 Iterations)')
axes[0].plot(noise_levels, mse_mlem10, 'go', label=f'MLEM (10 Iterations)')
axes[0].plot(noise_levels, mse_extra, 'ro', label=f'Extra-CNN ({10} Iterations)')
axes[0].set_xlabel('Relative Noise (%)')
axes[0].set_ylabel('MSE')
axes[0].set_title('MSE vs Noise Level')

# SSIM plot
##axes[1].plot(noise_levels, ssim_scores_cnn, 'bo', label=f'Intra-CNN ({num_its} Iterations)')
#axes[1].plot(noise_levels, ssim_scores_mlem20, 'go', label=f'MLEM (20 Iterations)')
axes[1].plot(noise_levels, ssim_mlem10, 'go', label=f'MLEM (10 Iterations)')
axes[1].plot(noise_levels, ssim_extra, 'ro', label=f'Extra-CNN ({10} Iterations)')
axes[1].set_xlabel('Relative Noise (%)')
axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM vs Noise Level')


# Unified legend above both plots
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.93]) 
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_noise_test_levels.png", dpi=300, bbox_inches='tight')
plt.show()

## Whole test set ###

mse_dist_extra = {noise: [] for noise in noise_levels}
ssim_dist_extra = {noise: [] for noise in noise_levels}

mse_dist_mlem10 = {noise: [] for noise in noise_levels}
ssim_dist_mlem10 = {noise: [] for noise in noise_levels}

with torch.no_grad():
    for noise in noise_levels:
        mse_list_extra = []
        ssim_list_extra = []
        mse_list_mlem10 = []
        ssim_list_mlem10 = []

        for true_image_np, sino_clean, av_count in zip(images_test, sinos_clean, av_counts):

            if noise is None:
                noisy_sino = sino_clean
            else:
                scale = scal_func(noise, av_count)
                noisy_sino = add_poisson(sino_clean, scale)

            # MLEM RECON 
            recon_mlem = mlem_reco(noisy_sino, theta, 10)

            # ExtraCNN DENOISING 
            inp = torch.from_numpy(recon_mlem).float().unsqueeze(0).unsqueeze(0).to(device)
            out = model(inp)
            recon = out.squeeze().cpu().numpy()

            # METRICS 
            mse_list_extra.append(mean_squared_error(true_image_np, recon))
            ssim_list_extra.append(ssim(true_image_np, recon,
                                  data_range=true_image_np.max() - true_image_np.min()))

            mse_list_mlem10.append(mean_squared_error(true_image_np, recon_mlem))
            ssim_list_mlem10.append(ssim(true_image_np, recon_mlem,
                                  data_range=true_image_np.max() - true_image_np.min()))

        # STORE DISTRIBUTIONS
        mse_dist_extra[noise] = mse_list_extra
        ssim_dist_extra[noise] = ssim_list_extra

        mse_dist_mlem10[noise] = mse_list_mlem10
        ssim_dist_mlem10[noise] = ssim_list_mlem10


fig, axes = plt.subplots(len(noise_levels), 2, figsize=(10, 4*len(noise_levels)), squeeze=False)

for i, noise in enumerate(noise_levels):

    # MSE histogram
    axes[i, 0].hist(mse_dist_extra[noise], bins=40, histtype='step', label='Extra-CNN')
    axes[i, 0].hist(mse_dist_mlem10[noise], bins=40, histtype='step', label='MLEM 10')
    axes[i, 0].set_title(f"MSE distribution (Noise={noise}%)")
    axes[i, 0].set_xlabel('MSE')
    axes[i, 0].set_ylabel('Count')
    axes[i, 0].legend()

    # SSIM histogram
    axes[i, 1].hist(ssim_dist_extra[noise], bins=40, histtype='step', label='Extra-CNN')
    axes[i, 1].hist(ssim_dist_mlem10[noise], bins=40, histtype='step', label='MLEM 10')
    axes[i, 1].set_title(f"SSIM distribution (Noise={noise}%)")
    axes[i, 1].set_xlabel('SSIM')
    axes[i, 1].set_ylabel('Count')
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/whole-noise-test-set.png",
            dpi=300, bbox_inches='tight')
plt.show()



