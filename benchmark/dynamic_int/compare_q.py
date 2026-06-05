import numpy as np
from triqs.gf import *

L = 0.1
w0 = 0.1 # User said w=0, but script has w0=0.1
beta = 20.0

q_tau_hyb = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=4096)
q_iw_hyb = make_gf_from_fourier(q_tau_hyb)  
q_iw_hyb << Function(lambda w: 2 * L/w0 * w0**2 / (w**2 - w0**2))
q_tau_hyb << Fourier(q_iw_hyb)

q_tau_seg = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=3999)
q_iw_seg = make_gf_from_fourier(q_tau_seg)  
q_iw_seg << Function(lambda w: 2 * L/w0 * w0**2 / (w**2 - w0**2))
q_tau_seg << Fourier(q_iw_seg)

print(f"CTHYB q_tau mean: {np.mean(q_tau_hyb.data[:, 0, 0]).real}")
print(f"CTSEG q_tau mean: {np.mean(q_tau_seg.data[:, 0, 0]).real}")
