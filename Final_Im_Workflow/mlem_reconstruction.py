import numpy as np
from skimage.transform import resize, radon, iradon
import gc
from utils.mlem import mlem_reco_its


'''for it in range(1, 50):

    #Setup
    nxd = 182
    theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

    X_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_train.npy")
    mlem_train = []

    for sino in X_train:
        mlem_train.append(mlem_reco(sino, theta, it))

    del(X_train)
    gc.collect()

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem{it}_train.npy", np.array(mlem_train, dtype=np.float32))
    del(mlem_train)
    gc.collect()

    X_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_val.npy")
    mlem_val = []

    for sino in X_val:
        mlem_val.append(mlem_reco(sino, theta, it))

    del(X_val)
    gc.collect()

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem{it}_val.npy", np.array(mlem_val, dtype=np.float32))
    del(mlem_val)
    gc.collect()

    X_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_test.npy")
    mlem_test = []

    for sino in X_test:
        mlem_test.append(mlem_reco(sino, theta, it))

    del(X_test)
    gc.collect()

    np.save(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/mlem{it}_test.npy", np.array(mlem_test, dtype=np.float32))
    del(mlem_test)
    gc.collect()'''

nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

base_path = "/scratch/bggjem001/pet_datasets/datasets"

X_train = np.load(f"{base_path}/sinograms_train.npy")
X_val   = np.load(f"{base_path}/sinograms_val.npy")
X_test  = np.load(f"{base_path}/sinograms_test.npy")

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
    np.save(f"{base_path}/mlem{it}_train.npy", current_train)
    gc.collect()

    # Val
    for i, sino in enumerate(X_val):
        current_val[i] = mlem_reco_its(sino, theta, steps, initial_guess=current_val[i])
    np.save(f"{base_path}/mlem{it}_val.npy", current_val)
    gc.collect()

    # Test
    for i, sino in enumerate(X_test):
        current_test[i] = mlem_reco_its(sino, theta, steps, initial_guess=current_test[i])
    np.save(f"{base_path}/mlem{it}_test.npy", current_test)
    gc.collect()

    prev_it = it
    print(f"Saved iteration {it}")

print("All MLEM reconstructions complete.")