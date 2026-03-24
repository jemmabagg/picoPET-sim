import numpy as np 
import matplotlib.pyplot as plt
import os 
import pandas as pd
import uproot

#Get cwd
cwd = os.getcwd()

file = uproot.open(f"{cwd}/Final_Im_Workflow/datasets/output_vereos_1.root")
print(file.keys())

tree = file["Singles4;1"]
print(tree.keys())

branches = ["PostPosition_X", "PostPosition_Y", "PostPosition_Z", "TotalEnergyDeposit", "PreStepUniqueVolumeID", "GlobalTime", "EventID"]

df = tree.arrays(branches, library="pd")

plt.hist(df["TotalEnergyDeposit"], bins=500)
plt.xlim(0, 1.0)
plt.savefig(f"{cwd}/Final_Im_Workflow/Plots/scatter_energies.png", dpi=300, bbox_inches="tight")

plt.show()
    