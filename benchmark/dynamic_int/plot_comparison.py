import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from h5 import HDFArchive
from triqs.gf import *

# Set up matplotlib configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rc('font', size=12)

data_loc = "/home/andrewhardy/Documents/Data/CTHYB_Data"
f_cthyb = f"{data_loc}/spin_spin_cthyb_lambda--0.1-U-4.0_b-10.0_nw-1000000_mins-150.h5"
f_ctseg = f"{data_loc}/spin_spin_ctseg_lambda--0.1-U-4.0_b-10.0.h5"

with HDFArchive(f_cthyb, "r") as A:
    G_tau_hyb = A["G_tau"]
    Sz_hyb = A["O_tau"]

with HDFArchive(f_ctseg, "r") as A:
    G_tau_seg = A["G_tau"]
    nn_seg = A["nn_tau"]

# Plot G_tau
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Extract meshes and data
tau_hyb_g = np.array([float(t) for t in G_tau_hyb.mesh])
g_hyb_data = G_tau_hyb["up"].data[:, 0, 0].real

tau_seg_g = np.array([float(t) for t in G_tau_seg.mesh])
g_seg_data = G_tau_seg["up"].data[:, 0, 0].real

plt.plot(tau_hyb_g, g_hyb_data, color="orange", linewidth=2.0, label="CTHYB")
plt.plot(tau_seg_g, g_seg_data, color="blue", linewidth=2.0, label="CTSEG", linestyle="--")
plt.title(r"$G_\uparrow(\tau)$ Comparison")
plt.ylabel(r"$G(\tau)$")
plt.xlabel(r"$\tau$")
plt.legend()

# Plot SzSz
plt.subplot(1, 2, 2)
SzSz_seg = 0.25 * (nn_seg['up', 'up'].data[:, 0, 0].real
                 + nn_seg['down', 'down'].data[:, 0, 0].real
                 - nn_seg['up', 'down'].data[:, 0, 0].real
                 - nn_seg['down', 'up'].data[:, 0, 0].real)

tau_seg = np.array([float(t) for t in nn_seg['up', 'up'].mesh])

tau_hyb = np.array([float(t) for t in Sz_hyb.mesh])
SzSz_hyb = Sz_hyb.data.real

plt.plot(tau_hyb, SzSz_hyb, color="orange", linewidth=2.0, label="CTHYB")
plt.plot(tau_seg, SzSz_seg, color="blue", linewidth=2.0, label="CTSEG", linestyle="--")
plt.title(r"$\langle S_z(\tau) S_z(0) \rangle$ Comparison")
plt.ylabel(r"$\langle S_z(\tau) S_z(0) \rangle$")
plt.xlabel(r"$\tau$")
plt.legend()

plt.tight_layout()
plt.savefig('comparison_plot.png')
print("Saved comparison_plot.png")
