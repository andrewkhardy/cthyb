from triqs.gf import *
import argparse
import triqs.utility.mpi as mpi
from triqs.gf.descriptors import Function
from triqs.gf.tools import *
from triqs.gf.block_gf import *
from triqs.operators import n
import h5
from triqs_ctseg import Solver
import numpy as np

beta = 10.0
U = 4.0
L = 1.0
n_tau = 8192
n_tau_bosonic = 8191
n_iw = 2048
h_int = U*n("up", 0)*n("down", 0)
gf_struct = [('down', 1), ('up', 1)]
constr_params = {"gf_struct": gf_struct, "beta": beta, "n_tau": n_tau, "n_tau_bosonic": n_tau_bosonic}
S = Solver(**constr_params)

Delta = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
g0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
q_tau = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=n_tau_bosonic)
q_iw = make_gf_from_fourier(q_tau)  
q_iw << Function(lambda w: 2 * L/0.1 * 0.1**2 / (w**2 - 0.1**2))
q_tau << Fourier(q_iw)
Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[q_tau, q_tau], [q_tau, q_tau]])
Q_iw = make_gf_from_fourier(Q_tau)
g0 << SemiCircular(2*1.0)
ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
zero_freq = np.where(np.abs(ivn) < 1e-10)
mu = U/2 - np.real((Q_iw["up", "up"].data[zero_freq][0,0,0]+Q_iw["up", "down"].data[zero_freq][0,0,0])/2.0)
Delta << iOmega_n + mu - inverse(g0)
S.Delta_tau << Fourier(Delta)
S.D0_tau << Q_tau

solve_params = {"h_int": h_int, "h_loc0": -mu * (n("up", 0) + n("down", 0)), "length_cycle": 100, "n_warmup_cycles": 100, "n_cycles": 1000, "measure_F_tau": False, "measure_nn_tau": True, "measure_nn_nu": True, "measure_nn_static": True, "measure_pert_order": True}
S.solve(**solve_params)

print("Type of nn_tau:", type(S.results.nn_tau))
print("Type of nn_nu:", type(S.results.nn_nu))
try:
    with h5.HDFArchive("test_out.h5", "w") as A:
        A['nn_tau'] = S.results.nn_tau
except Exception as e:
    import traceback
    traceback.print_exc()

