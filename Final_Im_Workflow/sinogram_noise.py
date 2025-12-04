import numpy as np
from utils.noise_fcn import add_poisson, scal_func, gaussian
from utils.image_ops import normalise
from scipy.optimize import curve_fit
import random

sinograms_clean_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_train.npy")
sinograms_clean_val   = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_val.npy")
sinograms_clean_test  = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_test.npy")

rel_noises = [None, 0.5, 1.0, 1.5, 2.0]

# Adding noise function
def add_noise(sinograms):

    output = []

    for sino in sinograms:
    
        noise_choice = random.choice(rel_noises)

        if noise_choice is None:
            output.append(sino)
        else:
            counts_bin = sino.ravel()
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

            av_counts = popt[0]
            #print(av_counts)
            scale = scal_func(noise_choice, av_counts)
            sino_noisy = add_poisson(sino, scale)
            output.append(sino_noisy)

    output = np.array(output)

    return output


sinograms_noisy_train = add_noise(sinograms_clean_train)
sinograms_noisy_val   = add_noise(sinograms_clean_val)
sinograms_noisy_test  = add_noise(sinograms_clean_test)

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy2_sinograms_train.npy", sinograms_noisy_train)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy2_sinograms_val.npy", sinograms_noisy_val)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy2_sinograms_test.npy", sinograms_noisy_test)
