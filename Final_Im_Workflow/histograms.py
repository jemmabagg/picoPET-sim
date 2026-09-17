import numpy as np
import matplotlib.pyplot as plt

# --- Normalisation functions ---

def normalise(img):
    """Min-max normalisation (the original, problematic one)."""
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def sum_normalise(img, target_sum=1.0):
    """Sum normalisation (the fix)."""
    img = img.astype(np.float32)
    total = img.sum()
    if total <= 0:
        return img
    return img * (target_sum / total)


# --- Setup ---

base = "/scratch/bggjem001/pet_datasets"
it = 9       # MLEM iteration count
idx = 0       # which test image to plot — pick one with clear structure
noise_high = 3.0  # high noise level for contrast

# --- Load data ---

gt = np.load(f"{base}/datasets/images_test.npy")[idx]
mlem_clean = np.load(f"{base}/datasets/mlem{it}_test.npy")[idx]
mlem_noisy = np.load(f"{base}/datasets/noisy{noise_high}_mlem{it}_test.npy")[idx]

# Optional: sinograms, if you have them saved. Comment out if not available.
sino_clean = np.load(f"{base}/datasets/sinograms_test.npy")[idx]
sino_noisy = np.load(f"{base}/datasets/noisy_sinograms_test{noise_high}.npy")[idx]
HAVE_SINOGRAMS = True  # set to True if sino_clean and sino_noisy are loaded above


# --- Numerical summary table ---

print(f"\n{'':25s} {'max':>10s} {'99th pct':>10s} {'median':>10s} {'mean':>10s}")
print("-" * 70)
for label, img in [("Ground truth", gt),
                   ("MLEM clean", mlem_clean),
                   (f"MLEM noisy ({noise_high}%)", mlem_noisy)]:
    print(f"{label:25s} "
          f"{img.max():>10.4f} "
          f"{np.percentile(img, 99):>10.4f} "
          f"{np.median(img):>10.4f} "
          f"{img.mean():>10.4f}")
print()


# --- Plot 1: the reconstruction stage (the core of the argument) ---

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Row 1: raw pixel value histograms
axes[0, 0].hist(gt.ravel(), bins=100, color='black')
axes[0, 0].set_title("Ground truth (raw)")
axes[0, 0].set_xlabel("Pixel value")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_yscale('log')

axes[0, 1].hist(mlem_clean.ravel(), bins=100, color='blue')
axes[0, 1].axvline(mlem_clean.max(), color='red', linestyle='--',
                   label=f'max = {mlem_clean.max():.2f}')
axes[0, 1].set_title("MLEM, clean input (raw)")
axes[0, 1].set_xlabel("Pixel value")
axes[0, 1].set_yscale('log')
axes[0, 1].legend()

axes[0, 2].hist(mlem_noisy.ravel(), bins=100, color='red')
axes[0, 2].axvline(mlem_noisy.max(), color='red', linestyle='--',
                   label=f'max = {mlem_noisy.max():.2f}')
axes[0, 2].set_title(f"MLEM, {noise_high}% noise input (raw)")
axes[0, 2].set_xlabel("Pixel value")
axes[0, 2].set_yscale('log')
axes[0, 2].legend()

# Row 2: after min-max normalisation
axes[1, 0].hist(normalise(gt).ravel(), bins=100, color='black')
axes[1, 0].set_title("Ground truth (min-max normalised)")
axes[1, 0].set_xlabel("Normalised pixel value")
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_yscale('log')

axes[1, 1].hist(normalise(mlem_clean).ravel(), bins=100, color='blue')
axes[1, 1].set_title("MLEM clean (min-max normalised)")
axes[1, 1].set_xlabel("Normalised pixel value")
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_yscale('log')

axes[1, 2].hist(normalise(mlem_noisy).ravel(), bins=100, color='red')
axes[1, 2].set_title("MLEM noisy (min-max normalised)")
axes[1, 2].set_xlabel("Normalised pixel value")
axes[1, 2].set_xlim(0, 1)
axes[1, 2].set_yscale('log')

plt.tight_layout()
plt.savefig(f"{base}/Plots/diagnostic_minmax_problem.png",
            dpi=300, bbox_inches='tight')
plt.show()


# --- Plot 2: sum normalisation fixes the comparison ---

gt_sn = sum_normalise(gt)
mlem_clean_sn = sum_normalise(mlem_clean)
mlem_noisy_sn = sum_normalise(mlem_noisy)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(gt_sn.ravel(), bins=100, color='black')
axes[0].set_title("Ground truth (sum normalised)")
axes[0].set_xlabel("Normalised pixel value")
axes[0].set_ylabel("Count")
axes[0].set_yscale('log')

axes[1].hist(mlem_clean_sn.ravel(), bins=100, color='blue')
axes[1].set_title("MLEM clean (sum normalised)")
axes[1].set_xlabel("Normalised pixel value")
axes[1].set_yscale('log')

axes[2].hist(mlem_noisy_sn.ravel(), bins=100, color='red')
axes[2].set_title(f"MLEM {noise_high}% noise (sum normalised)")
axes[2].set_xlabel("Normalised pixel value")
axes[2].set_yscale('log')

# Force same x-axis range for fair comparison
xmax = max(gt_sn.max(), mlem_clean_sn.max(), mlem_noisy_sn.max())
for ax in axes:
    ax.set_xlim(0, xmax)

plt.tight_layout()
plt.savefig(f"{base}/Plots/diagnostic_sumnorm_fix.png",
            dpi=300, bbox_inches='tight')
plt.show()


# --- Plot 3: sinogram stage (optional) ---

if HAVE_SINOGRAMS:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(sino_clean.ravel(), bins=100, color='blue')
    axes[0].set_title("Sinogram, no noise")
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale('log')

    axes[1].hist(sino_noisy.ravel(), bins=100, color='red')
    axes[1].set_title(f"Sinogram, {noise_high}% noise added")
    axes[1].set_xlabel("Pixel value")
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"{base}/Plots/diagnostic_sinogram_noise.png",
                dpi=300, bbox_inches='tight')
    plt.show()