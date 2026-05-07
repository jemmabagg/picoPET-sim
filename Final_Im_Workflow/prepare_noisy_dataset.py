import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import nibabel as nib
from utils.image_ops import crop_to_square, normalise
from scipy.optimize import curve_fit
from skimage.transform import radon, iradon
import random
from utils.noise_fcn import scal_func, add_poisson

#Functions
def gaussian(x, amplitude, mean, standard_deviation):
    return amplitude * np.exp ( - (x - mean)**2 / (2 * standard_deviation **2))

base = "/scratch/bggjem001/pet_datasets/datasets"

sets = ['train', 'val', 'test']

rel_noises = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

for noise in rel_noises:

    noisy_sinograms = {}

    for end in sets:  

        noisy_sinograms_list = []        
        clean_sinos = np.load(f"{base}/sinograms_{end}.npy") 

        for sino in clean_sinos:
            
            counts_bin = sino.ravel()
            bin_heights, bin_edges = np.histogram(counts_bin, bins=50)
            bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

            # Clean/weight the data (avoid zeros/NaNs and use Poisson weights)
            mask = np.isfinite(bin_heights) & np.isfinite(bin_centres) & (bin_heights > 0)
            x = bin_centres[mask]
            y = bin_heights[mask]
            sigma_w = np.sqrt(y)            # Poisson stdev
            sigma_w[sigma_w == 0] = 1.0     # safety
            
            # Initial guesses
            A0 = y.max()
            # weighted mean & std as starting points
            mu0 = np.average(x, weights=y)
            sigma0 = np.sqrt(np.average((x - mu0)**2, weights=y))
            # fallback if sigma0 is tiny/NaN
            if not np.isfinite(sigma0) or sigma0 <= 0:
                sigma0 = (x.max() - x.min())/6.0

            p0 = [A0, mu0, sigma0]

            lower = [0, x.min(), 1e-9]      # A>=0, mu in data range, sigma>0
            upper = [np.inf, x.max(), np.inf]

            popt, pcov = curve_fit(
                gaussian, x, y,
                p0=p0,
                sigma=sigma_w,
                absolute_sigma=True,   # interpret sigma as true stdevs
                bounds=(lower, upper), 
                maxfev=20000           # more function evaluations
                )

            av_counts = popt[0]
            #print(av_counts)
            scale = scal_func(noise, av_counts)
            sino_noisy = add_poisson(sino, scale)
            noisy_sinograms_list.append(sino_noisy)

        noisy_sinograms[end] = np.array(noisy_sinograms_list)


    # all three splits guaranteed to be from current noise level
    for end in sets:
        np.save(f"{base}/noisy_sinograms_{end}{noise}.npy", noisy_sinograms[end])