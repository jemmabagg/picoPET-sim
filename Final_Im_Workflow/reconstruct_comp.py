import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN

#Loading in the data 
images_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_test.npy")
mlem_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem_test.npy")
mlem35_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem35_test.npy")

### EXTRA-MLEM ###

extra_train_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_train_loss.npy")
extra_val_loss = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_val_loss.npy")

plt.plot(extra_train_loss, label="Train Loss")
plt.plot(extra_val_loss, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_loss_plot.png", dpi=300, bbox_inches='tight')
plt.show()

test_dataset = MLEMDataset(mlem_test, images_test)
print(f"Test: {len(test_dataset)}")

extra_test_loader = DataLoader(test_dataset, batch_size=1)

#Setup 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

extra_mlem = extraCNN().to(device)
extra_mlem.load_state_dict(torch.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extracnn.pth", map_location=device))
extra_mlem.eval()

extra_mses = []
extra_ssims = []
mlem_cnn_extra_ims = []
examples = []

with torch.no_grad():
    for idx, (rec_im, target) in enumerate(extra_test_loader):

        rec_im_np = rec_im.squeeze().numpy()
        target_np = target.squeeze().numpy()
       
        inp = torch.from_numpy(rec_im_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        out = extra_mlem(inp)                                    # [1,1,H,W]
        mlem_cnn_extra = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

        mse_extra_val = mean_squared_error(target_np, mlem_cnn_extra)
        ssim_val_denoise = ssim(target_np, mlem_cnn_extra, data_range=target_np.max() - target_np.min())

        extra_mses.append(mse_extra_val)
        extra_ssims.append(ssim_val_denoise)
        mlem_cnn_extra_ims.append(mlem_cnn_extra)

        # Save examples for visualisation
        if idx < 2:
            examples.append({
            "idx": idx,
            "target": target_np,
            "extra": mlem_cnn_extra,
            "extra_mse": mse_extra_val,
            "extra_ssim": ssim_val_denoise
        })

print(f"MLEM + CNN  - MSE: {np.mean(extra_mses):.4f}, SSIM: {np.mean(extra_ssims):.4f}")

### NORMAL MLEM ###

mlem_mses, mlem_ssims, mlem35_mses, mlem35_ssims = [], [], [], []

for idx, (mlem_im, im, mlem35_im) in enumerate(zip(mlem_test, images_test, mlem35_test)):
    mlem_mses.append(mean_squared_error(im, mlem_im))
    mlem_ssims.append(ssim(im, mlem_im, data_range=im.max() - im.min()))

    mlem35_mses.append(mean_squared_error(im, mlem35_im))
    mlem35_ssims.append(ssim(im, mlem35_im, data_range=im.max() - im.min()))

    # Save examples for visualisation
    if idx < 2:
        examples[idx]["mlem"] = mlem_im
        examples[idx]["mlem_mse"] = mlem_mses[idx]
        examples[idx]["mlem_ssim"] = mlem_ssims[idx]
        examples[idx]["mlem35"] = mlem35_im
        examples[idx]["mlem35_mse"] = mlem35_mses[idx]
        examples[idx]["mlem35_ssim"] = mlem35_ssims[idx]

for idx, (im, mlem35_im) in enumerate(zip(images_test, mlem35_test)):

    mlem35_mses.append(mean_squared_error(im, mlem35_im))
    mlem35_ssims.append(ssim(im, mlem35_im, data_range=im.max() - im.min()))

    # Save examples for visualisation
    if idx < 2:
        examples[idx]["mlem35"] = mlem35_im
        examples[idx]["mlem35_mse"] = mlem35_mses[idx]
        examples[idx]["mlem35_ssim"] = mlem35_ssims[idx]

### STANDARD OSEM ###

osem_standard_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/osem_test_standard.npy")

osem_standard_mses, osem_standard_ssims = [], []

for idx, (osem_standard_im, im) in enumerate(zip(osem_standard_test, images_test)):
    osem_standard_mses.append(mean_squared_error(im, osem_standard_im))
    osem_standard_ssims.append(ssim(im, osem_standard_im, data_range=im.max() - im.min()))

    # Save examples for visualisation
    if idx < 2:
        examples[idx]["osem_standard"] = osem_standard_im
        examples[idx]["osem_standard_mse"] = osem_standard_mses[idx]
        examples[idx]["osem_standard_ssim"] = osem_standard_ssims[idx]

### Equivalient OSEM ###

osem_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/osem_test.npy")

osem_mses, osem_ssims = [], []

for idx, (osem_im, im) in enumerate(zip(osem_test, images_test)):
    osem_mses.append(mean_squared_error(im, osem_im))
    osem_ssims.append(ssim(im, osem_im, data_range=im.max() - im.min()))

    # Save examples for visualisation
    if idx < 2:
        examples[idx]["osem"] = osem_im
        examples[idx]["osem_mse"] = osem_mses[idx]
        examples[idx]["osem_ssim"] = osem_ssims[idx]

### FINAL COMPARISON PLOTS ###
n_examples = len(examples)

fig, axs = plt.subplots(
    n_examples, 6,
    figsize=(14, 3.2 * n_examples),
    squeeze=False
)

# --- column titles ---
col_titles = [
    "Ground Truth",
    "Extra-CNN",
    "MLEM (10 Its)",
    "OSEM (3 Its, 3 Subsets)",
    "MLEM (35 Its)",
    "OSEM (8 Its, 4 Subsets)"
]

for j, title in enumerate(col_titles):
    axs[0, j].set_title(title, fontsize=13, pad=12)

# --- plot images + metrics ---
for i, ex in enumerate(examples):
    
    imgs = [
        ex["target"],
        ex["extra"],
        ex["mlem"],
        ex["osem"],
        ex["mlem35"],
        ex["osem_standard"]
        
    ]
    
    # Display each reconstruction
    for j, img in enumerate(imgs):
        axs[i, j].imshow(img, cmap="hot")
        axs[i, j].axis("off")

    # --- Add MSE + SSIM under each reconstruction ---
    text_kwargs = dict(ha='center', va='top', fontsize=9)

    # Extra-CNN
    axs[i, 1].text(
        0.5, -0.10,
        f"MSE: {ex['extra_mse']:.4f}\nSSIM: {ex['extra_ssim']:.4f}",
        transform=axs[i,1].transAxes, **text_kwargs
    )

    # MLEM (10 Iterations)
    axs[i, 2].text(
        0.5, -0.10,
        f"MSE: {ex['mlem_mse']:.4f}\nSSIM: {ex['mlem_ssim']:.4f}",
        transform=axs[i,2].transAxes, **text_kwargs
    )

    # OSEM 3 its
    axs[i, 3].text(
        0.5, -0.10,
        f"MSE: {ex['osem_mse']:.4f}\nSSIM: {ex['osem_ssim']:.4f}",
        transform=axs[i,3].transAxes, **text_kwargs
    )

    # MLEM (35 Iterations)
    axs[i, 5].text(
        0.5, -0.10,
        f"MSE: {ex['mlem35_mse']:.4f}\nSSIM: {ex['mlem35_ssim']:.4f}",
        transform=axs[i,5].transAxes, **text_kwargs
    )

    # OSEM standard (8 its, 4 subsets)
    axs[i, 4].text(
        0.5, -0.10,
        f"MSE: {ex['osem_standard_mse']:.4f}\nSSIM: {ex['osem_standard_ssim']:.4f}",
        transform=axs[i,4].transAxes, **text_kwargs
    )


# Make the layout tighter but leave space for titles 
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

plt.savefig(
    "/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_comparison_clean.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Combine both MSE arrays
all_mse = np.concatenate([extra_mses, mlem_mses, osem_mses])
mse_bins = np.linspace(all_mse.min(), all_mse.max(), 40)

#Combine SSIM arrays
all_ssim = np.concatenate([extra_ssims, mlem_ssims, osem_ssims])
ssim_bins = np.linspace(all_ssim.min(), all_ssim.max(), 40)

fig = plt.figure(figsize=(12,4))

# MSE Distribution 
ax1 = plt.subplot(1,2,1)
ax1.hist(extra_mses, bins=mse_bins, histtype='step', label="Extra-CNN")
ax1.hist(mlem_mses, bins=mse_bins, histtype='step', label="MLEM")
ax1.hist(osem_mses, bins=mse_bins, histtype='step', label="OSEM")
ax1.hist(osem_standard_mses, bins=mse_bins, histtype='step', label="Standard OSEM")
ax1.hist(mlem35_mses, bins=mse_bins, histtype='step', label="Standard MLEM")
ax1.set_title('Distribution of MSE (Test Set)')
ax1.set_xlabel('MSE Value')
ax1.set_ylabel('Number of Images')

# SSIM Distribution 
ax2 = plt.subplot(1,2,2)
ax2.hist(extra_ssims, bins=ssim_bins, histtype='step', label="Extra-CNN")
ax2.hist(mlem_ssims, bins=ssim_bins, histtype='step', label="MLEM")
ax2.hist(osem_ssims, bins=ssim_bins, histtype='step', label="OSEM")
ax2.hist(osem_standard_ssims, bins=ssim_bins, histtype='step', label="Standard OSEM")
ax2.hist(mlem35_ssims, bins=ssim_bins, histtype='step', label="Standard MLEM")
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

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_av_mse_ssim.png",
            dpi=300, bbox_inches='tight')
plt.show()
