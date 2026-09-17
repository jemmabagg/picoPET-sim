import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from models.mlem_dataset import MLEMDataset
from models.extramlem import extraCNN
from utils.image_ops import normalise, sum_normalise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base = "/scratch/bggjem001/pet_datasets"

# Loading test images
test_ims_sn = np.load(f"{base}/datasets/images_test_sn.npy")
n_ims = test_ims_sn.shape[0]
scale = 182 * 182

# Getting the MLEM iterations
save_every = 3
iterations = list(range(save_every, 37, save_every))
noise_levels = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

# Noise levels at which to save qualitative comparison figures
qualitative_noise_levels = [0.5, 3.0]  # 5% and 30%

# Indices of test images to show in qualitative figure
qualitative_example_idx = [0, 1]

global_dr = float(test_ims_sn.max() - test_ims_sn.min())

# Storage for summary table
summary_rows = []

# Storage for qualitative figure data
# keyed by (noise_level): {'mlem_peak_it': int, 'mlem_images': np.array, 'extra_images': np.array}
qualitative_data = {}


def compute_metrics(target, recon, data_range):
    """Return per-image arrays of SSIM, NMSE, PSNR."""
    n = target.shape[0]
    ssim_v = np.zeros(n)
    nmse_v = np.zeros(n)
    psnr_v = np.zeros(n)
    for i in range(n):
        ssim_v[i] = ssim(target[i], recon[i], data_range=data_range)
        nmse_v[i] = mean_squared_error(target[i], recon[i]) / np.var(target[i])
        psnr_v[i] = psnr(target[i], recon[i], data_range=data_range)
    return ssim_v, nmse_v, psnr_v


for noise in noise_levels:

    prefix = "clean" if noise == 0 else f"noisy{noise}"
    mlem_prefix = '' if noise == 0 else f"noisy{noise}_"

    # Per-iteration means
    ssim_mlem_mean, ssim_extra_mean = [], []
    psnr_mlem_mean, psnr_extra_mean = [], []
    nmse_mlem_mean, nmse_extra_mean = [], []

    # Per-iteration stds (across test set)
    ssim_mlem_std, ssim_extra_std = [], []
    psnr_mlem_std, psnr_extra_std = [], []
    nmse_mlem_std, nmse_extra_std = [], []

    # Keep MLEM and extra-CNN reconstructions for qualitative figure
    save_qualitative = (noise in qualitative_noise_levels)
    per_iteration_mlem = {} if save_qualitative else None
    per_iteration_extra = {} if save_qualitative else None

    for it in iterations:
        print(f"Noise {noise}, iteration {it}")

        # Loading MLEM test images
        mlem_test_raw = np.load(f"{base}/datasets/{mlem_prefix}mlem{it}_test.npy")
        mlem_test_sn = np.array([sum_normalise(im) for im in mlem_test_raw])
        mlem_test_scaled = mlem_test_sn * scale

        # Loading the trained extra-CNN model
        model = extraCNN().to(device)
        model.load_state_dict(torch.load(
            f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/{prefix}_extra{it}_sn1.pth",
            map_location=device
        ))
        model.eval()

        with torch.no_grad():
            enhanced = []
            for im in mlem_test_scaled:
                x = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().to(device)
                out = model(x)
                enhanced.append(out.squeeze().cpu().numpy())

        enhanced = np.array(enhanced) / scale

        # Metrics
        s_m, n_m, p_m = compute_metrics(test_ims_sn, mlem_test_sn, global_dr)
        s_e, n_e, p_e = compute_metrics(test_ims_sn, enhanced, global_dr)

        ssim_mlem_mean.append(s_m.mean()); ssim_mlem_std.append(s_m.std())
        nmse_mlem_mean.append(n_m.mean()); nmse_mlem_std.append(n_m.std())
        psnr_mlem_mean.append(p_m.mean()); psnr_mlem_std.append(p_m.std())

        ssim_extra_mean.append(s_e.mean()); ssim_extra_std.append(s_e.std())
        nmse_extra_mean.append(n_e.mean()); nmse_extra_std.append(n_e.std())
        psnr_extra_mean.append(p_e.mean()); psnr_extra_std.append(p_e.std())

        if save_qualitative:
            per_iteration_mlem[it] = mlem_test_sn[qualitative_example_idx].copy()
            per_iteration_extra[it] = enhanced[qualitative_example_idx].copy()

        del mlem_test_raw, mlem_test_sn, enhanced, mlem_test_scaled, model
        torch.cuda.empty_cache()

    # Convert to arrays
    ssim_mlem_mean = np.array(ssim_mlem_mean); ssim_extra_mean = np.array(ssim_extra_mean)
    nmse_mlem_mean = np.array(nmse_mlem_mean); nmse_extra_mean = np.array(nmse_extra_mean)
    psnr_mlem_mean = np.array(psnr_mlem_mean); psnr_extra_mean = np.array(psnr_extra_mean)
    ssim_mlem_std = np.array(ssim_mlem_std); ssim_extra_std = np.array(ssim_extra_std)
    iters_arr = np.array(iterations)

    # ---- Summary table values ----

    # MLEM peak SSIM (semi-convergence point)
    mlem_ssim_peak_idx = ssim_mlem_mean.argmax()
    mlem_ssim_peak_it = iters_arr[mlem_ssim_peak_idx]
    mlem_ssim_peak = ssim_mlem_mean[mlem_ssim_peak_idx]
    mlem_ssim_peak_std = ssim_mlem_std[mlem_ssim_peak_idx]

    mlem_psnr_peak_idx = psnr_mlem_mean.argmax()
    mlem_psnr_peak_it = iters_arr[mlem_psnr_peak_idx]
    mlem_psnr_peak = psnr_mlem_mean[mlem_psnr_peak_idx]

    mlem_nmse_min_idx = nmse_mlem_mean.argmin()
    mlem_nmse_min_it = iters_arr[mlem_nmse_min_idx]
    mlem_nmse_min = nmse_mlem_mean[mlem_nmse_min_idx]

    # extra-CNN converged values (mean over last 3 iterations once plateaued)
    extra_ssim_conv = ssim_extra_mean[-3:].mean()
    extra_ssim_conv_std = ssim_extra_std[-3:].mean()
    extra_psnr_conv = psnr_extra_mean[-3:].mean()
    extra_nmse_conv = nmse_extra_mean[-3:].mean()

    # First iteration crossing SSIM = 0.9
    def first_cross(arr, thresh, iters):
        idx = np.where(arr >= thresh)[0]
        return int(iters[idx[0]]) if len(idx) > 0 else None

    mlem_cross_09 = first_cross(ssim_mlem_mean, 0.9, iters_arr)
    extra_cross_09 = first_cross(ssim_extra_mean, 0.9, iters_arr)

    # Relative improvement of extra-CNN converged vs MLEM peak
    ssim_rel = extra_ssim_conv / mlem_ssim_peak
    nmse_rel = extra_nmse_conv / mlem_nmse_min  # < 1 means extra-CNN better
    psnr_rel = extra_psnr_conv / mlem_psnr_peak

    summary_rows.append({
        'noise_pct': noise * 10,
        'mlem_ssim_peak': mlem_ssim_peak,
        'mlem_ssim_peak_std': mlem_ssim_peak_std,
        'mlem_ssim_peak_it': mlem_ssim_peak_it,
        'mlem_psnr_peak': mlem_psnr_peak,
        'mlem_psnr_peak_it': mlem_psnr_peak_it,
        'mlem_nmse_min': mlem_nmse_min,
        'mlem_nmse_min_it': mlem_nmse_min_it,
        'extra_ssim_conv': extra_ssim_conv,
        'extra_ssim_conv_std': extra_ssim_conv_std,
        'extra_psnr_conv': extra_psnr_conv,
        'extra_nmse_conv': extra_nmse_conv,
        'mlem_cross_ssim_09': mlem_cross_09,
        'extra_cross_ssim_09': extra_cross_09,
        'ssim_rel_improvement': ssim_rel,
        'nmse_rel_improvement': nmse_rel,
        'psnr_rel_improvement': psnr_rel,
    })

    # Save qualitative data:
    #   - MLEM and extra-CNN at the same (final) iteration count
    #   - MLEM at its peak-SSIM iteration as a separate reference
    if save_qualitative:
        comp_it = int(iters_arr[-1])
        qualitative_data[noise] = {
            'comp_it': comp_it,
            'mlem_comp_images': per_iteration_mlem[comp_it],
            'extra_comp_images': per_iteration_extra[comp_it],
            'mlem_peak_it': int(mlem_ssim_peak_it),
            'mlem_peak_images': per_iteration_mlem[int(mlem_ssim_peak_it)],
        }

    # ---- Per-noise iteration plot (existing behaviour, kept) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].errorbar(iters_arr, ssim_mlem_mean, yerr=ssim_mlem_std, fmt='bo', label="MLEM", capsize=3)
    axes[0].errorbar(iters_arr, ssim_extra_mean, yerr=ssim_extra_std, fmt='ro', label="extra-CNN", capsize=3)
    axes[0].set_xlabel("MLEM Iterations"); axes[0].set_ylabel("SSIM"); axes[0].legend()

    axes[1].errorbar(iters_arr, nmse_mlem_mean, yerr=nmse_mlem_std, fmt='bo', label="MLEM", capsize=3)
    axes[1].errorbar(iters_arr, nmse_extra_mean, yerr=nmse_extra_std, fmt='ro', label="extra-CNN", capsize=3)
    axes[1].set_xlabel("MLEM Iterations"); axes[1].set_ylabel("NMSE"); axes[1].legend()

    axes[2].errorbar(iters_arr, psnr_mlem_mean, yerr=psnr_mlem_std, fmt='bo', label="MLEM", capsize=3)
    axes[2].errorbar(iters_arr, psnr_extra_mean, yerr=psnr_extra_std, fmt='ro', label="extra-CNN", capsize=3)
    axes[2].set_xlabel("MLEM Iterations"); axes[2].set_ylabel("PSNR"); axes[2].legend()

    plt.suptitle(f"Performance Metrics vs MLEM Iteration (rel noise = {noise*10}%)")
    plt.tight_layout()
    plt.savefig(f"{base}/Plots/metrics_vs_iterations_{noise*10}_sn1.png", dpi=300, bbox_inches='tight')
    plt.close()


# ---- Build and save summary table ----

df = pd.DataFrame(summary_rows)
print("\n=== Summary table ===")
print(df.to_string(index=False))

df.to_csv(f"{base}/Plots/summary_table.csv", index=False)

# Also write a LaTeX-formatted version with rounded values
df_latex = df.copy()
for col in ['mlem_ssim_peak', 'mlem_ssim_peak_std', 'extra_ssim_conv', 'extra_ssim_conv_std',
            'ssim_rel_improvement', 'nmse_rel_improvement', 'psnr_rel_improvement']:
    df_latex[col] = df_latex[col].round(4)
for col in ['mlem_psnr_peak', 'extra_psnr_conv']:
    df_latex[col] = df_latex[col].round(2)
for col in ['mlem_nmse_min', 'extra_nmse_conv']:
    df_latex[col] = df_latex[col].round(4)

with open(f"{base}/Plots/summary_table.tex", "w") as f:
    f.write(df_latex.to_latex(index=False, float_format="%.4f"))


# ---- Qualitative comparison figure ----
# Columns: Ground truth | MLEM (comp_it) | extra-CNN (comp_it) | MLEM @ max SSIM
# Metrics for the displayed examples are quoted in each panel title, computed
# with the same definitions as the summary table (compute_metrics, global_dr).

for noise, data in qualitative_data.items():
    n_examples = len(qualitative_example_idx)
    fig, axes = plt.subplots(n_examples, 4, figsize=(16, 4.5 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    comp_it = data['comp_it']
    peak_it = data['mlem_peak_it']

    for row, im_idx in enumerate(qualitative_example_idx):
        gt = test_ims_sn[im_idx]
        mlem_comp = data['mlem_comp_images'][row]
        extra_comp = data['extra_comp_images'][row]
        mlem_peak = data['mlem_peak_images'][row]

        # Per-image metrics (single-element batch reuses compute_metrics exactly)
        s_mc, n_mc, p_mc = (v[0] for v in compute_metrics(gt[None], mlem_comp[None], global_dr))
        s_ec, n_ec, p_ec = (v[0] for v in compute_metrics(gt[None], extra_comp[None], global_dr))
        s_mp, n_mp, p_mp = (v[0] for v in compute_metrics(gt[None], mlem_peak[None], global_dr))

        # Shared intensity scale across all reconstruction panels
        vmax_recon = max(gt.max(), mlem_comp.max(), extra_comp.max(), mlem_peak.max())

        axes[row, 0].imshow(gt, cmap='hot', vmin=0, vmax=vmax_recon)
        axes[row, 0].set_title("Ground Truth")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mlem_comp, cmap='hot', vmin=0, vmax=vmax_recon)
        axes[row, 1].set_title(
            f"MLEM ({comp_it} its)\n"
            f"SSIM {s_mc:.3f}  NMSE {n_mc:.3f}  PSNR {p_mc:.1f}")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(extra_comp, cmap='hot', vmin=0, vmax=vmax_recon)
        axes[row, 2].set_title(
            f"extra-CNN ({comp_it} its)\n"
            f"SSIM {s_ec:.3f}  NMSE {n_ec:.3f}  PSNR {p_ec:.1f}")
        axes[row, 2].axis('off')

        axes[row, 3].imshow(mlem_peak, cmap='hot', vmin=0, vmax=vmax_recon)
        axes[row, 3].set_title(
            f"MLEM @ max SSIM ({peak_it} its)\n"
            f"SSIM {s_mp:.3f}  NMSE {n_mp:.3f}  PSNR {p_mp:.1f}")
        axes[row, 3].axis('off')

    plt.suptitle(f"Reconstruction comparison (rel noise = {noise*10:.0f}%)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{base}/Plots/qualitative_comparison_{noise*10:.0f}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

print("\nDone. Summary table saved to summary_table.csv and summary_table.tex")
print("Qualitative figures saved for noise levels:", list(qualitative_data.keys()))