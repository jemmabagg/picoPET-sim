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

#Loading in the data 
images_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_test.npy")
mlem_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_mlem_test.npy")

#Normalise the reconstructed images
mlem_test = np.array([normalise(im) for im in mlem_test])
images_test = np.array([normalise(im) for im in images_test])

### EXTRA-MLEM ###

extra_train_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_denoise_train_loss.npy")
extra_val_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_denoise_val_loss.npy")

plt.plot(extra_train_loss, label="Train Loss")
plt.plot(extra_val_loss, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/denoised_extra_loss_plot.png", dpi=300, bbox_inches='tight')
plt.show()

test_dataset = MLEMDataset(mlem_test, images_test)
print(f"Test: {len(test_dataset)}")

extra_test_loader = DataLoader(test_dataset, batch_size=10)

#Setup 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

extra_mlem = extraCNN().to(device)
extra_mlem.load_state_dict(torch.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_denoise.pth", map_location=device))
extra_mlem.eval()

extra_mses = []
extra_ssims = []
mlem_cnn_extra_ims = []
examples = []

with torch.no_grad():
    for idx, (rec_im, target) in enumerate(extra_test_loader):

        # rec_im: [B, 1, H, W]
        # target: [B, 1, H, W]

        rec_im = rec_im.to(device).float()
        target = target.to(device).float()

        out = extra_mlem(rec_im)             # out: [B, 1, H, W]

        # Convert to numpy
        out_np = out.squeeze(1).cpu().numpy()     # [B, H, W]
        target_np = target.squeeze(1).cpu().numpy()

        # Loop over items in the batch
        for b in range(out_np.shape[0]):
            mse_val = mean_squared_error(target_np[b], out_np[b])
            ssim_val = ssim(target_np[b], out_np[b], 
                            data_range=target_np[b].max() - target_np[b].min())

            extra_mses.append(mse_val)
            extra_ssims.append(ssim_val)

            if len(examples) < 2:   # save first 2 examples
                examples.append({
                    "idx": idx * extra_test_loader.batch_size + b,
                    "target": target_np[b],
                    "extra": out_np[b],
                    "extra_mse": mse_val,
                    "extra_ssim": ssim_val
                })

print(f"MLEM + CNN  - MSE: {np.mean(extra_mses):.4f}, SSIM: {np.mean(extra_ssims):.4f}")

# Combine both MSE arrays
all_mse = np.concatenate([extra_mses])
mse_bins = np.linspace(all_mse.min(), all_mse.max(), 40)

#Combine SSIM arrays
all_ssim = np.concatenate([extra_ssims])
ssim_bins = np.linspace(all_ssim.min(), all_ssim.max(), 40)

fig = plt.figure(figsize=(12,4))

# MSE Distribution 
ax1 = plt.subplot(1,2,1)
ax1.hist(extra_mses, bins=mse_bins, histtype='step')
ax1.set_title('Distribution of MSE (Test Set)')
ax1.set_xlabel('MSE Value')
ax1.set_ylabel('Number of Images')

# SSIM Distribution 
ax2 = plt.subplot(1,2,2)
ax2.hist(extra_ssims, bins=ssim_bins, histtype='step')
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

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/denoised_extra_av_mse_ssim.png",
            dpi=300, bbox_inches='tight')
plt.show()