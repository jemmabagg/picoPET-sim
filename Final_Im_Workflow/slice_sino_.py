import uproot
import numpy as np
import matplotlib.pyplot as plt

base = "/scratch/bggjem001/picoPET-sim/Final_Im_Workflow"

file = uproot.open(f"{base}/datasets/phillips_hadded_coin.root")
file.keys ()

tree = file["Coincidences"]
print(tree.keys())

branches = [
    "PostPosition_X1", "PostPosition_Y1", "PostPosition_Z1",
    "PostPosition_X2", "PostPosition_Y2", "PostPosition_Z2",
    "s", "theta"
]

dz = 2.0 #mm
Nz = 74
z_min = -Nz * dz / 2  
z_max =  Nz * dz / 2  
z_edges = np.linspace(z_min, z_max, Nz + 1)


Ntheta = 180
theta_edges = np.linspace(0, np.pi, Ntheta + 1)

s_max = 125
Ns = 125
s_edges = np.linspace(-s_max, s_max, Ns + 1 )

#edges are arrays that define the boundaries for the sinogram bins 
#working with discrete data

sinograms = np.zeros((Nz, Ns, Ntheta), dtype=np.int32)

for arrays in tree.iterate(branches, library="np", step_size="200 MB"):

    z_mid = (arrays["PostPosition_Z1"] + arrays["PostPosition_Z2"]) / 2

    #k = np.floor((z_mid - z_min) / dz).astype(int)
    k = np.searchsorted(z_edges, z_mid, side='right') - 1
    valid = (k >= 0) & (k < Nz)

    '''phi = arrays["theta"][valid]
    x1 = arrays["PostPosition_X1"][valid]
    x2 = arrays["PostPosition_X2"][valid]
    y1 = arrays["PostPosition_Y1"][valid]
    y2 = arrays["PostPosition_Y2"][valid]'''
    s = arrays['s'][valid]
    theta = arrays['theta'][valid]
    k  = k[valid]

    '''mY = (y1+y2)/2
    mX = (x1+x2)/2
    theta = phi+np.pi/2
    s = mX * np.cos(theta) + mY * np.sin(theta)'''

    #np.searchsorted finds which bin the s/t values belongs in 
    s_bin = np.searchsorted(s_edges, s, side='right') - 1
    t_bin = np.searchsorted(theta_edges, theta, side='right') - 1

    good = (
        (s_bin >= 0) & (s_bin < Ns) &
        (t_bin >= 0) & (t_bin < Ntheta)
    )

    np.add.at(sinograms, (k[good], s_bin[good], t_bin[good]), 1)

sinogram_zsum = sinograms.sum(axis=0)
theta_edges_deg = np.rad2deg(theta_edges)

print("Number of z slices:", sinograms.shape[0])

plt.figure(figsize=(8, 6))

plt.imshow(
    sinogram_zsum,
    extent=[
        theta_edges_deg[0], theta_edges_deg[-1],  # x: theta
        s_edges[0], s_edges[-1]                    # y: s
    ],
    aspect="auto",
    origin="lower"
)

plt.xlabel("theta [deg]")
plt.ylabel("s [mm]")
plt.colorbar(label="Counts")

plt.title("Sinogram integrated over z")
plt.tight_layout()
plt.savefig("/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/phillips_sinogram_combined.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))

k_rand = 33

plt.imshow(
    sinograms[k_rand],
    extent=[
        theta_edges_deg[0], theta_edges_deg[-1],
        s_edges[0], s_edges[-1]
    ],
    aspect="auto",
    origin="lower"
)

plt.xlabel("theta [deg]")
plt.ylabel("s [mm]")
plt.colorbar(label="Counts")
plt.title(f"Sinogram at z-slice k = {k_rand}")

plt.tight_layout()
plt.savefig(
    f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/Plots/phillips_sinogram_slice_k{k_rand}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

np.save(f"{base}/datasets/phillips_sliced_sino.npy", sinograms)




