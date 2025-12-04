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
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp ( - (x - mean)**2 / (2 * standard_deviation **2))

#Setup
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

##Reading in the NRM2018 dataset

#Loading the files 
paths = [
    "/scratch/bggjem001/picoPET-sim/PET_data/LondonPride_000101_dynPET.img",
    "/scratch/bggjem001/picoPET-sim/PET_data/LondonPride_000102_dynPET.img",
    "/scratch/bggjem001/picoPET-sim/PET_data/LondonPride_000103_dynPET.img",
    "/scratch/bggjem001/picoPET-sim/PET_data/LondonPride_000104_dynPET.img",
    "/scratch/bggjem001/picoPET-sim/PET_data/LondonPride_000105_dynPET.img",
]

#Data has shape (182, 218, 182, 23), so we have 23 complete 3D PET volumes 
#4 dimenion is the time frame because it is a dynamic PET scan

#Defining how many z-slices and t-slices we want to use in our dataset
z_half_window = 65
t_half_window = 5
rel_noises = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

images = []
sinograms = []

for p in paths: 

    img = nib.load(p)
    pet_data = img.get_fdata(dtype=np.float32)

    #Getting the full z-range and the start and end point based on our range
    total_slices = pet_data.shape[2]
    centre = total_slices // 2
    z_start = centre - z_half_window
    z_end = centre + z_half_window

    #Getting the full t-range and the start and end point based on our range
    total_slices = pet_data.shape[3]
    centre = total_slices // 2
    t_start = centre - t_half_window
    t_end = centre + t_half_window

    for t in range(t_start, t_end): #time frams

        for z in range(z_start, z_end): #z frames 

            slice_img = pet_data[:, :, z, t]

            image_cropped = crop_to_square(slice_img)
            image_resized = resize(image_cropped, (nxd,nxd), anti_aliasing=True)

            #Normalise image
            image_normalised = normalise(image_resized)

            images.append(image_normalised)

            #Generate Sinogram by forward projecting image
            sino = radon(image_normalised, theta=theta, circle=True)

            #noise_choice = random.choice(rel_noises)
            noise_choice = 0.05

            if noise_choice is None:
                sinograms.append(sino)
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
                sinograms.append(sino_noisy)


images = np.array(images)
sinograms = np.array(sinograms)

# Split into train, val, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(sinograms, images, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_images_train.npy", y_train)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_images_val.npy", y_val)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_images_test.npy", y_test)

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_sinograms_train.npy", X_train)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_sinograms_val.npy", X_val)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/noisy_sinograms_test.npy", X_test)