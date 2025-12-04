import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

#Setup
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

def osem_reco(sino, azi_angles, num_its=3, num_subsets=3):
   
    image_size = sino.shape[0]
    osem_rec = np.ones((image_size, image_size))
    
    # Split the angles and sinogram into subsets
    subsets = np.array_split(np.arange(len(azi_angles)), num_subsets)

    for it in range(num_its):
        
        for subset_idx, subset in enumerate(subsets):
            # Extract subset of sinogram and angles
            sino_subset = sino[:, subset]
            angles_subset = azi_angles[subset]

            # Compute sensitivity for this subset: A_m^T * 1
            sens_subset = iradon(
                np.ones_like(sino_subset),
                angles_subset,
                circle=True,
                filter_name=None,
                output_size=image_size
            )

            # Forward project current estimate (A_m x)
            fp_subset = radon(osem_rec, angles_subset, circle=True)

            # Ratio y_m / (A_m x)
            ratio = sino_subset / (fp_subset + 1e-6)

            # Backproject ratio (A_m^T (y_m / A_m x))
            corr_subset = iradon(
                ratio,
                angles_subset,
                circle=True,
                filter_name=None,
                output_size=image_size
            )

            # Update image estimate
            osem_rec *= corr_subset / (sens_subset + 1e-6)

    return osem_rec

#Loading in sinograms
sinograms = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_test.npy")
osem_ims = []

for sino in sinograms:
    osem_ims.append(osem_reco(sino, theta))

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/osem_test.npy", osem_ims)
