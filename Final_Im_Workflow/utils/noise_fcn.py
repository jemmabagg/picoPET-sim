import numpy as np

def add_poisson(sino, scale):
    scaled_sino = sino * scale
    scaled_sino = np.clip(scaled_sino, 0, 1e6)

    #Generate Poisson distributed random values with mean = scaled_sino
    noisy = np.random.poisson(scaled_sino)
    return noisy / scale 

def scal_func(n, av_counts):
    r = n / 100
    x = 1 / ( (r**2) * (av_counts) )

    return x

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp ( - (x - mean)**2 / (2 * standard_deviation **2))