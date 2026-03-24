import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_sinogram(coincidences_df: pd.DataFrame,
                  bins_theta: int = 180, bins_r: int = 180):
    """
    Plot the sinogram of coincidence pairs.
    
    For each coincidence pair, the sinogram parameters (theta, r) are calculated.
    Here, theta (in radians) is the angle of the normal to the line-of-response (LOR),
    and r is the perpendicular distance from the origin to the LOR.
    """

    #Plotting the TOF 3D Sinogram
    thetas = []
    rs = []
    
    # Calculate sinogram parameters for each coincidence pair.
    for _, row in coincidences_df.iterrows():
        x1 = row["PostPosition_X_1"]
        y1 = row["PostPosition_Y_1"]
        x2 = row["PostPosition_X_2"]
        y2 = row["PostPosition_Y_2"]
        
        # Midpoint of the LOR
        xm = (x1 + x2) / 2.0
        ym = (y1 + y2) / 2.0
        
        # Orientation of the LOR.
        phi = np.arctan2(y2 - y1, x2 - x1)
        
        # Sinogram angle: the normal's angle (wrapped to [0, pi)).
        theta = (phi + np.pi/2) % np.pi
        # Perpendicular distance from the origin.
        r = xm * np.cos(theta) + ym * np.sin(theta)
        
        thetas.append(theta)
        rs.append(r)

    # Convert lists to NumPy arrays
    thetas = np.array(np.degrees(thetas))
    rs = np.array(rs)

    # 2D histogram
    sinogram, r_edges, theta_edges = np.histogram2d(rs, thetas, bins=[bins_r, bins_theta])

    print("Sinogram Shape:", sinogram.shape)

    return sinogram, r_edges, theta_edges