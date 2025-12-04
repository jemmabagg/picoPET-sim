import numpy as np

#Defining function to crop an image to a square
def crop_to_square(img):
    h, w = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2

    return img[top:top + min_dim, left:left + min_dim]

#Defining function to normalise image
def normalise(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)