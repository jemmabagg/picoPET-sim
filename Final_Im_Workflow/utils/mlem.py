import numpy as np
from skimage.transform import radon, iradon

#Defining the basic MLEM reconstruction function
def mlem_reco(sino, azi_angles, num_its: int):

    #Starting with the initial guess
    image_size = sino.shape[0]
    mlem_rec = np.ones((image_size, image_size)) #initial guess of a uniform image

    #Sensitivity image
    sens_image = iradon(np.ones_like(sino), azi_angles, circle=True, filter_name=None, output_size=image_size)

    #MLEM Loop
    for it in range(num_its):
        
        #Forward projection of iamge estimate
        fp = radon(mlem_rec, azi_angles, circle=True)

        #Ratio
        ratio = sino / (fp + 0.000001)

        #Backprojecting to get the correction factor
        correction = iradon(ratio, azi_angles, circle=True, filter_name=None, output_size=image_size) 

        #Normalising by sensitivity image
        norm = correction / (sens_image + 0.000001)

        mlem_rec = mlem_rec * norm

    return mlem_rec

def mlem_reco_its(sino, azi_angles, num_its: int, initial_guess=None):

    image_size = sino.shape[0]
    
    # Use provided initial guess or default to ones
    mlem_rec = initial_guess if initial_guess is not None else np.ones((image_size, image_size))

    sens_image = iradon(np.ones_like(sino), azi_angles, circle=True, filter_name=None, output_size=image_size)

    for it in range(num_its):
        fp = radon(mlem_rec, azi_angles, circle=True)
        ratio = sino / (fp + 1e-6)
        correction = iradon(ratio, azi_angles, circle=True, filter_name=None, output_size=image_size)
        norm = correction / (sens_image + 1e-6)
        mlem_rec = mlem_rec * norm

    return mlem_rec
