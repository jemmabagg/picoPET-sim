import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN

#Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## EXTRA-CNN ###

images = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_test.npy")
mlem_images = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem_test.npy")

test_dataset = MLEMDataset(mlem_images, images)

#Loading in the data
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
results = {}

for alpha in alphas:
    results[alpha] = {
        "train_loss": np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_train_loss_alpha_{alpha}.npy"),
        "train_mse": np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_train_mse_alpha_{alpha}.npy"),
        "train_ssim": np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_train_ssim_alpha_{alpha}.npy"),
        "val_loss": np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_val_loss_alpha_{alpha}.npy"),
        "val_mse": np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_val_mse_alpha_{alpha}.npy"),
        "val_ssim": np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extra_val_ssim_alpha_{alpha}.npy"),
    }

plt.figure(figsize=(10,6))

for alpha, history in results.items():
    plt.plot(history["train_loss"], label=f"Train α={alpha}")

plt.title("Training & Validation Loss for Different α Values")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_alpha_loss_plot.png", dpi=300, bbox_inches='tight')

# Evaluating the test set over different alphas 

results_eval = {alpha: {"mse": [], "ssim": []} for alpha in alphas}

for alpha in alphas:
    print(f"Evaluating model alpha = {alpha}")

    # Load model
    model = extraCNN().to(device)
    model.load_state_dict(torch.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/extra_combined/extramlem_alpha_{alpha}.pth"))
    model.eval()

    for test_idx in range(len(test_dataset)):
        sinogram, true_image_np = test_dataset[test_idx]
        sinogram_np = sinogram.numpy()
        true_image_np = true_image_np.numpy()

        with torch.no_grad():

            #DeepPET
            inp = torch.from_numpy(sinogram_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
            out = model(inp)                                    # [1,1,H,W]
            image_dp = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
       
        mse_val = mean_squared_error(true_image_np, image_dp)
        ssim_val = ssim(true_image_np, image_dp, data_range=true_image_np.max() - true_image_np.min())

        results_eval[alpha]["mse"].append(mse_val)
        results_eval[alpha]["ssim"].append(ssim_val)

plt.figure(figsize=(10,4))

#MSE Distribution 
ax1 = plt.subplot(1,2,1)
for alpha in alphas:
    ax1.hist(results_eval[alpha]["mse"], bins=30, histtype='step', label=f'α = {alpha}')
ax1.set_title('Distribution of MSE (Test Set)')
ax1.set_xlabel('MSE Value')
ax1.set_ylabel('Number of Images')

#SSIM Distribution 
ax2 = plt.subplot(1,2,2)
for alpha in alphas:
    ax2.hist(results_eval[alpha]["ssim"], bins=30, histtype='step', label=f'α = {alpha}')
ax2.set_title('Distribution of SSIM (Test Set)')
ax2.set_xlabel('SSIM Value')
ax2.set_ylabel('Number of Images')

#Shared Legend
handles, labels = ax1.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper center', ncol=len(alphas))

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_alpha_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Compute mean values per alpha
mean_mse = [np.mean(results_eval[a]["mse"]) for a in alphas]
mean_ssim = [np.mean(results_eval[a]["ssim"]) for a in alphas]

plt.figure(figsize=(10,4))

#Mean MSE vs Alpha 
ax1 = plt.subplot(1,2,1)
ax1.plot(alphas, mean_mse, 'o', color='steelblue')
ax1.set_title('Mean MSE vs Alpha')
ax1.set_xlabel('Alpha (α)')
ax1.set_ylabel('Mean MSE')


#Mean SSIM vs Alpha 
ax2 = plt.subplot(1,2,2)
ax2.plot(alphas, mean_ssim, 'o', color='darkorange')
ax2.set_title('Mean SSIM vs Alpha')
ax2.set_xlabel('Alpha (α)')
ax2.set_ylabel('Mean SSIM')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/extra_alpha_means.png", dpi=300, bbox_inches='tight')
plt.show()
