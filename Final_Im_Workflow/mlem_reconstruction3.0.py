import numpy as np
from skimage.transform import resize, radon, iradon
import gc
from utils.mlem import mlem_reco_its

nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

base_path = "/home/bggjem001/sinograms"

X_train = np.load(f"{base_path}/noisy_sinograms_train3.0.npy")
X_val   = np.load(f"{base_path}/noisy_sinograms_val3.0.npy")
X_test  = np.load(f"{base_path}/noisy_sinograms_test3.0.npy")

print("Sinograms loaded")

save_every = 3  

base_path = "/scratch/bggjem001/pet_datasets/datasets"

# Initialise current reconstructions as ones (same as mlem_reco default)
current_train = np.ones((X_train.shape[0], nxd, nxd), dtype=np.float32)
current_val   = np.ones((X_val.shape[0],   nxd, nxd), dtype=np.float32)
current_test  = np.ones((X_test.shape[0],  nxd, nxd), dtype=np.float32)

prev_it = 0

for it in range(save_every, 37, save_every):  # 3, 6, 9, ..., 36

    steps = it - prev_it  # only run NEW iterations (always 3)
    print(f"Running MLEM iterations {prev_it+1} to {it}")

    # Train
    for i, sino in enumerate(X_train):
        current_train[i] = mlem_reco_its(sino, theta, steps, initial_guess=current_train[i])
    np.save(f"{base_path}/noisy3.0_mlem{it}_train.npy", current_train)
    gc.collect()

    # Val
    for i, sino in enumerate(X_val):
        current_val[i] = mlem_reco_its(sino, theta, steps, initial_guess=current_val[i])
    np.save(f"{base_path}/noisy3.0_mlem{it}_val.npy", current_val)
    gc.collect()

    # Test
    for i, sino in enumerate(X_test):
        current_test[i] = mlem_reco_its(sino, theta, steps, initial_guess=current_test[i])
    np.save(f"{base_path}/noisy3.0_mlem{it}_test.npy", current_test)
    gc.collect()

    prev_it = it
    print(f"Saved iteration {it}")

print("All MLEM reconstructions complete.")