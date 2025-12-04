import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN
from scipy.optimize import curve_fit
from utils.noise_fcn import scal_func, add_poisson, gaussian

#Loading in the data 
sinograms_clean_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_test.npy")
sinograms_clean_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_train.npy")
sinograms_clean_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_val.npy")

sinograms_noisy_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy2_sinograms_test.npy")
sinograms_noisy_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy2_sinograms_train.npy")
sinograms_noisy_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy2_sinograms_val.npy")

### Sinogram Denoiser

extra_train_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/sino_denoise_train_loss.npy")
extra_val_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/sino_denoise_val_loss.npy")

plt.plot(extra_train_loss, label="Train Loss")
plt.plot(extra_val_loss, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/sino_denoise_loss_plot.png", dpi=300, bbox_inches='tight')
plt.show()

test_dataset = MLEMDataset(sinograms_noisy_test, sinograms_clean_test)
train_dataset = MLEMDataset(sinograms_noisy_train, sinograms_clean_train)
val_dataset = MLEMDataset(sinograms_noisy_val, sinograms_clean_val)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} Test: {len(test_dataset)}")

extra_test_loader = DataLoader(test_dataset, batch_size=1)
extra_train_loader = DataLoader(train_dataset, batch_size=1)
extra_val_loader = DataLoader(val_dataset, batch_size=1)

#Setup 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

extra_mlem = extraCNN().to(device)
extra_mlem.load_state_dict(torch.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/sino_denoise.pth", map_location=device))
extra_mlem.eval()

extra_mses = []
extra_ssims = []
mlem_cnn_extra_ims = []
examples = []

denoised_sinos_train, denoised_sinos_test, denoised_sinos_val = [], [], []

with torch.no_grad():
    for idx, (rec_im, target) in enumerate(extra_test_loader):

        rec_im_np = rec_im.squeeze().numpy()
        target_np = target.squeeze().numpy()
       
        inp = torch.from_numpy(rec_im_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        out = extra_mlem(inp)                                    # [1,1,H,W]
        mlem_cnn_extra = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

        denoised_sinos_test.append(mlem_cnn_extra)

        #print("rec_im_np:", rec_im_np.shape)
        #print("target_np:", target_np.shape)
        #print("inp:", inp.shape)
        #print("output:", mlem_cnn_extra.shape)

        mse_extra_val = mean_squared_error(target_np, mlem_cnn_extra)
        ssim_val_denoise = ssim(target_np, mlem_cnn_extra, data_range=target_np.max() - target_np.min())

        extra_mses.append(mse_extra_val)
        extra_ssims.append(ssim_val_denoise)
        mlem_cnn_extra_ims.append(mlem_cnn_extra)

with torch.no_grad():
    for idx, (rec_im, target) in enumerate(extra_train_loader):

        rec_im_np = rec_im.squeeze().numpy()
        target_np = target.squeeze().numpy()
       
        inp = torch.from_numpy(rec_im_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        out = extra_mlem(inp)                                    # [1,1,H,W]
        mlem_cnn_extra = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

        denoised_sinos_train.append(mlem_cnn_extra)


with torch.no_grad():
    for idx, (rec_im, target) in enumerate(extra_val_loader):

        rec_im_np = rec_im.squeeze().numpy()
        target_np = target.squeeze().numpy()
       
        inp = torch.from_numpy(rec_im_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        out = extra_mlem(inp)                                    # [1,1,H,W]
        mlem_cnn_extra = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

        denoised_sinos_val.append(mlem_cnn_extra)

denoised_sinos_test = np.array(denoised_sinos_test)
denoised_sinos_train = np.array(denoised_sinos_train)
denoised_sinos_val = np.array(denoised_sinos_val)

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_sinograms_train.npy", denoised_sinos_train)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_sinograms_val.npy", denoised_sinos_val)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_sinograms_test.npy", denoised_sinos_test)

# Combine both MSE arrays
all_mse = np.concatenate([extra_mses])
mse_bins = np.linspace(all_mse.min(), all_mse.max(), 40)

#Combine SSIM arrays
all_ssim = np.concatenate([extra_ssims])
ssim_bins = np.linspace(all_ssim.min(), all_ssim.max(), 40)

fig = plt.figure(figsize=(12,4))

# MSE Distribution 
ax1 = plt.subplot(1,2,1)
ax1.hist(extra_mses, bins=mse_bins, histtype='step', label="Sinogram Denoiser")
ax1.set_title('Distribution of MSE (Test Set)')
ax1.set_xlabel('MSE Value')
ax1.set_ylabel('Number of Images')

# SSIM Distribution 
ax2 = plt.subplot(1,2,2)
ax2.hist(extra_ssims, bins=ssim_bins, histtype='step', label="Sinogram Denoiser")
ax2.set_title('Distribution of SSIM (Test Set)')
ax2.set_xlabel('SSIM Value')
ax2.set_ylabel('Number of Images')

# Shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center',
           ncol=2,
           fontsize=12,
           frameon=False,
           bbox_to_anchor=(0.5, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/denoise_sino_av_mse_ssim.png",
            dpi=300, bbox_inches='tight')
plt.show()

##Seeing how it performs on one sinogram over varying noise levels

#Getting the average counts 

av_counts = []

for sino_clean in sinograms_clean_test:

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

idx = 50
sino = sinograms_clean_test[idx]
av_count = av_counts[idx]

noise_levels = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

mse_extra, ssim_extra = [], []
mse_before, mse_after = [], []
ssim_before, ssim_after = [], []

#Loading in the model
model = extraCNN().to(device)
model.load_state_dict(
    torch.load(
        "/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/sino_denoise.pth",
        map_location=device
    )
)
model.eval()

with torch.no_grad():
    for noise in noise_levels:

        if noise is None:
            noisy_sino = sino
        else:
            scale = scal_func(noise, av_count)
            noisy_sino = add_poisson(sino, scale)

        mse_before.append(mean_squared_error(sino, noisy_sino))
        ssim_before.append(ssim(sino, noisy_sino, data_range=sino.max() - sino.min()))

        inp = torch.from_numpy(noisy_sino).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        out = model(inp)                                    # [1,1,H,W]
        mlem_cnn_extra = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

        mse_extra_val = mean_squared_error(sino, mlem_cnn_extra)
        mse_after.append(mse_extra_val)
        ssim_val_denoise = ssim(sino, mlem_cnn_extra, data_range=sino.max() - sino.min())
        ssim_after.append(ssim_val_denoise)
        mse_extra.append(mse_extra_val)
        ssim_extra.append(ssim_val_denoise)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # <- use fig and axes

# MSE plot
axes[0].plot(noise_levels, mse_extra, 'ro', label=f'Sinogram Denoiser')
axes[0].set_xlabel('Relative Noise (%)')
axes[0].set_ylabel('MSE')
axes[0].set_title('MSE vs Noise Level')

# SSIM plot
axes[1].plot(noise_levels, ssim_extra, 'ro', label=f'Sinogram Denoiser')
axes[1].set_xlabel('Relative Noise (%)')
axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM vs Noise Level')


# Unified legend above both plots
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.93]) 
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/denoise_sino_noise_test_levels.png", dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---------------- MSE subplot ----------------
axes[0].plot(noise_levels, mse_before, 'bo', label="Noisy Sinogram")
axes[0].plot(noise_levels, mse_after,  'ro', label="Denoised Sinogram")
axes[0].set_xlabel("Noise Level (%)")
axes[0].set_ylabel("MSE")
axes[0].set_title("MSE vs Noise Level")
axes[0].legend()

# ---------------- SSIM subplot ----------------
axes[1].plot(noise_levels, ssim_before, 'bo', label="Noisy Sinogram")
axes[1].plot(noise_levels, ssim_after,  'ro', label="Denoised Sinogram")
axes[1].set_xlabel("Noise Level (%)")
axes[1].set_ylabel("SSIM")
axes[1].set_title("SSIM vs Noise Level")
axes[1].legend()

plt.tight_layout()
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/denoise_sino_compare.png",
            dpi=300, bbox_inches='tight')
plt.show()

