import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 
from skimage.transform import iradon
from utils.mlem import mlem_reco
from utils.sinograms import gen_sinogram

#Get the current working directory
cwd = os.getcwd()

df = pd.read_csv(f"{cwd}/Final_Im_Workflow/datasets/ListModeData_Simon2.0.csv")

print(df.head())
print("Number of coincidences =  " + str(df.shape[0]))


sinogram, r_edges, theta_edges = gen_sinogram(df, bins_theta=90, bins_r=90)
sinogram_reco = sinogram.T
theta = (theta_edges[:-1] + theta_edges[1:]) / 2

##FBP Reconstruction:

fbp_im = iradon(sinogram_reco, theta, filter_name='ramp', circle=False)

##MLEM Im
mlem_im = mlem_reco(sinogram_reco, theta, 50)

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# Sinogram
im0 = axes[0].imshow(sinogram, cmap="gray_r", origin="lower", aspect="auto",extent=[theta_edges[0], theta_edges[-1], r_edges[0], r_edges[-1]])
axes[0].set_title("Sinogram")
axes[0].set_xlabel("Theta (degrees)")
axes[0].set_ylabel("s (mm)")
fig.colorbar(im0, ax=axes[0])

# FBP
im1 = axes[1].imshow(fbp_im)
axes[1].set_title("FBP Reconstruction")
fig.colorbar(im1, ax=axes[1])

# MLEM
im2 = axes[2].imshow(mlem_im)
axes[2].set_title("MLEM Reconstruction (20 Its)")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()

plt.savefig(f"{cwd}/Final_Im_Workflow/Plots/simon_all_results.png", dpi=300, bbox_inches="tight")

plt.show()
    