import uproot
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from utils.sinograms import gen_sinogram

#Get cwd
cwd = os.getcwd()

file = uproot.open(f"{cwd}/Final_Im_Workflow/datasets/output_vereos_1.root")
print(file.keys())

tree = file["Singles4;1"]
print(tree.keys())

branches = ["PostPosition_X", "PostPosition_Y", "PostPosition_Z", "TotalEnergyDeposit", "PreStepUniqueVolumeID", "GlobalTime", "EventID"]

df = tree.arrays(branches, library="pd")
print(df["GlobalTime"].max())

### GETTING THE RANDOMS SINOGRAM USING THE DELAYED TIMING WINDOW METHOD ###

def delayed_window(df: pd.DataFrame, time_window: float) -> pd.DataFrame:

    #Sorting the df by GlobalTime to do the search
    df_sorted = df.sort_values("GlobalTime").reset_index(drop=True)
    times = df_sorted["GlobalTime"].values
    energies = df_sorted["TotalEnergyDeposit"].values
    n = len(df_sorted)
    rand_coincidences = []

    #Defining energy window based on Ryan's slide (350  < e < 650 keV)
    energy_lower = 0.350
    energy_upper = 0.650

    #Delayed timing shift (assuming window is 1.5ns)
    delay_shift = 10*time_window

    #Applying energy cut + timing cuts
    for i in range(n):

        delayed_centre = times[i] + delay_shift
        lower_time = delayed_centre - time_window
        upper_time = delayed_centre + time_window

        lower_idx = np.searchsorted(times, lower_time, side='left')
        upper_idx = np.searchsorted(times, upper_time, side='right')

        if energy_lower <= energies[i] <= energy_upper:

            for k in range(lower_idx, upper_idx):

                if energy_lower <= energies[k] <= energy_upper:
                    pair = {}

                    for col in df_sorted.columns:
                        pair[f"{col}_1"] = df_sorted.iloc[i][col]
                        pair[f"{col}_2"] = df_sorted.iloc[k][col]

                    rand_coincidences.append(pair)

    return pd.DataFrame(rand_coincidences)

rand_coincidences_df = delayed_window(df, time_window=1.5) #Global time stored in ns 
print(rand_coincidences_df.head())

sinogram, r_edges, theta_edges = gen_sinogram(rand_coincidences_df)

plt.imshow(sinogram, origin="lower", aspect="auto",extent=[theta_edges[0], theta_edges[-1], r_edges[0], r_edges[-1]])
plt.title("Sinogram")
plt.xlabel("Theta (degrees)")
plt.ylabel("s (mm)")
plt.colorbar()
plt.savefig(f"{cwd}/Final_Im_Workflow/Plots/randoms_sino.png", dpi=300, bbox_inches="tight")

plt.show()
    

