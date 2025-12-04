import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import nibabel as nib
from utils.image_ops import crop_to_square, normalise
from skimage.transform import resize, radon, iradon

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
z_half_window = 50
t_half_window = 5

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

            sinograms.append(sino)

images = np.array(images)
sinograms = np.array(sinograms)

# Split into train, val, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(sinograms, images, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_train.npy", y_train)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_val.npy", y_val)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/images_test.npy", y_test)

np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_train.npy", X_train)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_val.npy", X_val)
np.save("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/datasets/sinograms_test.npy", X_test)

