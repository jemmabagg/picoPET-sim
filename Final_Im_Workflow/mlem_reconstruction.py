import numpy as np
from skimage.transform import resize, radon, iradon
import gc
from utils.mlem import mlem_reco

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

base_path = "/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets"

X_train = np.load(f"{base_path}/sinograms_train.npy")
X_val   = np.load(f"{base_path}/sinograms_val.npy")
X_test  = np.load(f"{base_path}/sinograms_test.npy")

print("Sinograms loaded")

save_every = 3  

for it in range(1, 50):

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
