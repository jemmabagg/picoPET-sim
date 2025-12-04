import numpy as np
from skimage.transform import resize, radon, iradon
import gc
from utils.mlem import mlem_reco

#Setup
nxd = 182
theta = np.linspace(0., 180., max(nxd, 180), endpoint=False)

X_train = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_sinograms_train.npy")
mlem_train = []

for sino in X_train:
    mlem_train.append(mlem_reco(sino, theta, 10))

del(X_train)
gc.collect()

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_mlem_train.npy", np.array(mlem_train, dtype=np.float32))
del(mlem_train)
gc.collect()

X_val = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_sinograms_val.npy")
mlem_val = []

for sino in X_val:
    mlem_val.append(mlem_reco(sino, theta, 10))

del(X_val)
gc.collect()

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_mlem_val.npy", np.array(mlem_val, dtype=np.float32))
del(mlem_val)
gc.collect()

X_test = np.load("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_sinograms_test.npy")
mlem_test = []

for sino in X_test:
    mlem_test.append(mlem_reco(sino, theta, 35))

del(X_test)
gc.collect()

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/denoised_mlem_test.npy", np.array(mlem_test, dtype=np.float32))
del(mlem_test)
gc.collect()