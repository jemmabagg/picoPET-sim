import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import radon
from utils.mlem import mlem_reco
 
from utils.noise_fcn import gaussian, scal_func
 
 
# Config
base = "/scratch/bggjem001/pet_datasets"
sino_index = 0          # which sinogram from the training set to use
n_bins = 50             # histogram bins for counts-per-bin distribution
 
# Note on argument order: utils.noise_fcn.gaussian uses (x, mean, amplitude, std).
# curve_fit treats the order of p0/bounds as positional matches to that signature.
 
# Load one sinogram from the training split
clean_sinos = np.load(f"{base}/datasets/sinograms_train.npy")
sino = clean_sinos[sino_index]
 
'''# ----- Left plot: counts-per-bin distribution + Gaussian fit -----
 
counts_bin = sino.ravel()
bin_heights, bin_edges = np.histogram(counts_bin, bins=n_bins)
bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2
 
mask = np.isfinite(bin_heights) & np.isfinite(bin_centres) & (bin_heights > 0)
x = bin_centres[mask]
y = bin_heights[mask]
sigma_w = np.sqrt(y)
sigma_w[sigma_w == 0] = 1.0
 
A0 = y.max()
mu0 = np.average(x, weights=y)
sigma0 = np.sqrt(np.average((x - mu0) ** 2, weights=y))
if not np.isfinite(sigma0) or sigma0 <= 0:
    sigma0 = (x.max() - x.min()) / 6.0
 
# gaussian signature is (x, mean, amplitude, standard_deviation)
p0 = [mu0, A0, sigma0]
lower = [x.min(), 0, 1e-9]
upper = [x.max(), np.inf, np.inf]
 
popt, _ = curve_fit(
    gaussian, x, y,
    p0=p0,
    sigma=sigma_w,
    absolute_sigma=True,
    bounds=(lower, upper),
    maxfev=20000,
)
 
mu_fit, A_fit, sigma_fit = popt
N = mu_fit  # mean counts per bin (peak of Gaussian)
 
x_smooth = np.linspace(bin_centres.min(), bin_centres.max(), 500)
gauss_curve = gaussian(x_smooth, *popt)
 
# ----- Right plot: relative noise (%) vs scaling factor -----
 
# Analytic curve: rel_noise (%) = 100 / sqrt(N*x)
x_at_30 = 100.0**2 / (30.0**2 * N)
scales = np.linspace(x_at_30, 1.0, 500)
rel_noise_pct = 100.0 / np.sqrt(N * scales)
 
# Overlay points derived via scal_func: pick target noise levels (%),
# compute the corresponding scaling factor, and plot (x, target).
# If these sit on the analytic curve, the function is internally consistent.
target_noise_pct = np.array([3, 5, 10, 15, 20, 25, 30])
scales_from_func = np.array([scal_func(n, N) for n in target_noise_pct])
 
# ----- Plot -----
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
 
# Left
axes[0].hist(counts_bin, bins=n_bins, edgecolor="none")
axes[0].plot(x_smooth, gauss_curve, color="orange", linewidth=2, label="Gaussian fit")
axes[0].set_xlabel("Counts per bin")
axes[0].set_ylabel("Number of bins")
axes[0].legend()
 
# Right
axes[1].plot(scales, rel_noise_pct, color="tab:blue", linewidth=1.5,
             label="Analytic: $100/\\sqrt{Nx}$")
#axes[1].scatter(scales_from_func, target_noise_pct,
                #color="red", s=40, zorder=3, label="scal_func targets")
axes[1].set_xlabel("Scaling Factor")
axes[1].set_ylabel("Relative Noise (%)")
axes[1].set_ylim(0, 30)
axes[1].legend()

 
plt.tight_layout()
plt.savefig(f"{base}/Plots/fig_6_4_regenerated.png", dpi=200)
plt.show()
 
print(f"Fitted mean counts per bin (N): {N:.1f}")
#print(f"Baseline relative noise (unscaled, x=1): {baseline_pct:.2f}%")
print("\nscal_func verification:")
for n, x_val in zip(target_noise_pct, scales_from_func):
    achieved = 100.0 / np.sqrt(N * x_val)
    print(f"  target {n}% -> scale {x_val:.4f} -> achieved {achieved:.2f}%")'''

#Getting comparison plot

im_index = 33
theta = np.linspace(0., 180., max(182, 180), endpoint=False)

gt_images = np.load(f"{base}/datasets/images_train.npy")
gt_image = gt_images[im_index]

sino = radon(gt_image, theta=theta, circle=True)

mlem_image = mlem_reco(sino, theta, 100)

global_dr = float(gt_image.max() - gt_image.min())

nmse_val = mean_squared_error(gt_image, mlem_image) / np.var(gt_image)

ssim_val = ssim(gt_image, mlem_image, data_range=global_dr)

psnr_val = psnr(gt_image, mlem_image, data_range=global_dr)

plt.figure(figsize=(4,4))
plt.imshow(gt_image, cmap='hot')
plt.axis('off')
plt.title("Ground Truth Image")
plt.tight_layout()
plt.savefig(f"{base}/Plots/gt_image_{im_index}.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(4,4))
plt.imshow(mlem_image, cmap='hot')
plt.axis('off')
plt.title(f"MLEM (100 Iterations)\nNMSE: {nmse_val:.4f}  PSNR: {psnr_val:.4f}  SSIM: {ssim_val:.4f}")
plt.tight_layout()
plt.savefig(f"{base}/Plots/mlem_image_{im_index}.png", dpi=200, bbox_inches="tight")
plt.show()

angles = [30, 60, 90, 120, 180]
ssim_vals, nmse_vals, psnr_vals = [], [], []

for angle in angles:

    theta = np.linspace(0., 180., angle, endpoint=False)

    sino = radon(gt_image, theta=theta, circle=True)

    mlem_image = mlem_reco(sino, theta, 100)

    global_dr = float(gt_image.max() - gt_image.min())

    nmse_vals.append(mean_squared_error(gt_image, mlem_image) / np.var(gt_image))

    ssim_vals.append(ssim(gt_image, mlem_image, data_range=global_dr))

    psnr_vals.append(psnr(gt_image, mlem_image, data_range=global_dr))


#NMSE

plt.figure()
plt.scatter(angles, nmse_vals, color='red')
plt.xlabel("Number of Projection Angles")
plt.ylabel("NMSE Score")
plt.savefig(f"{base}/Plots/nmse_proj_angles.png", dpi=200, bbox_inches="tight")
plt.show()

#PSNR

plt.figure()
plt.scatter(angles, psnr_vals, color='green',)
plt.xlabel("Number of Projection Angles")
plt.ylabel("PSNR Score")
plt.savefig(f"{base}/Plots/psnr_proj_angles.png", dpi=200, bbox_inches="tight")
plt.show()

#SSIM

plt.figure()
plt.scatter(angles, ssim_vals)
plt.xlabel("Number of Projection Angles")
plt.ylabel("SSIM Score")
plt.savefig(f"{base}/Plots/ssim_proj_angles.png", dpi=200, bbox_inches="tight")
plt.show()