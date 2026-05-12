import h5py
import numpy as np
with h5py.File("/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_cthyb_J-1.0-U-4.0_0.0_b-10.0_nw-1000000_mins-50_lf=True_nl=10.h5", "r") as A:
    g_data = A["G_tau"]["HDFArchive_Data"]["up"]["data"]
    print("Last 5 points of G_tau:")
    for i in range(1, 6):
        print(g_data[-i, 0, 0].real)
