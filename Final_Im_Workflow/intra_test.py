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

        rec_im_np = rec_im.squeeze(0).numpy()
        target_np = target.squeeze(0).numpy()
       
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