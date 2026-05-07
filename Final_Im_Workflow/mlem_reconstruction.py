import numpy as np
from skimage.transform import resize, radon, iradon
import gc
from utils.mlem import mlem_reco

nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

base_path = "/scratch/bggjem001/pet_datasets/datasets"

X_train = np.load(f"{base_path}/sinograms_train.npy")
X_val   = np.load(f"{base_path}/sinograms_val.npy")
X_test  = np.load(f"{base_path}/sinograms_test.npy")

print("Sinograms loaded")

save_every = 3  

for it in range(1, 36):

    # Skip iterations we do not want to save
    if it % save_every != 0:
        continue

    print(f"Running MLEM iteration {it}")

    # Train
    mlem_train = np.empty((X_train.shape[0], nxd, nxd), dtype=np.float32)
    for i, sino in enumerate(X_train):
        mlem_train[i] = mlem_reco(sino, theta, it)

    np.save(f"{base_path}/mlem{it}_train.npy", mlem_train)
    del mlem_train
    gc.collect()

    # Val
    mlem_val = np.empty((X_val.shape[0], nxd, nxd), dtype=np.float32)
    for i, sino in enumerate(X_val):
        mlem_val[i] = mlem_reco(sino, theta, it)

    np.save(f"{base_path}/mlem{it}_val.npy", mlem_val)
    del mlem_val
    gc.collect()

    # Test
    mlem_test = np.empty((X_test.shape[0], nxd, nxd), dtype=np.float32)
    for i, sino in enumerate(X_test):
        mlem_test[i] = mlem_reco(sino, theta, it)

    np.save(f"{base_path}/mlem{it}_test.npy", mlem_test)
    del mlem_test
    gc.collect()

print("All MLEM reconstructions complete.")
