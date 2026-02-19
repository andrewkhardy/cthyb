# Single orbital with dynamical spin-spin interactions (CT-INT solver).
# Adapted from ctseg_spin_spin.py for benchmarking against CT-SEG and CT-HYB.
from triqs_ctint import Solver

import argparse
import triqs.utility.mpi as mpi
from triqs.gf import *
from triqs.gf.descriptors import Function
from triqs.gf.tools import *
from triqs.operators import n
from triqs.gf.block_gf import *
import h5
from triqs.utility.h5diff import h5diff

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spin-spin benchmarking (CT-INT).')
parser.add_argument('--U', type=float, default=4.0, help='U parameter')
parser.add_argument('--J', type=float, default=1.0, help='J parameter')
parser.add_argument('--i1', type=float, default=1.0,  help='i1 switch (0 or 1)')
parser.add_argument('--i2', type=float, default=1.0,  help='i2 switch (0 or 1)')
parser.add_argument('--i3', type=float, default=1.0,  help='i3 switch (0 or 1)')
parser.add_argument('--i4', type=float, default=1.0,  help='i4 switch (0 or 1)')
parser.add_argument('--i5', type=float, default=1.0,  help='i5 switch (0 or 1)')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
args, unknown = parser.parse_known_args()

# Numerical values
beta = args.beta
U = args.U
mu = U/2
J = args.J
i_1, i_2, i_3, i_4, i_5 = args.i1, args.i2, args.i3, args.i4, args.i5
n_tau = 4096
n_tau_bosonic = 2001
n_iw = 1025
# Solver construction parameters
block_names = ['dn','up']
gf_struct = [(bl, 1) for bl in block_names]
h_int = U * n(block_names[0],0)*n(block_names[1],0)

n_iw = 1025

S = Solver(beta = beta,
               gf_struct = gf_struct,
               n_iw = n_iw,
               n_tau = n_tau,
               use_D = True,
               use_Jperp = True,
               n_tau_dynamical_interactions = n_tau_bosonic,
               n_iw_dynamical_interactions = n_tau_bosonic//2)

# Get inputs from reference file
with h5.HDFArchive("ctint.ref.h5", 'r') as Af:
    g0 = Af["dmft_loop/i_001/S/G0_iw/up"]
    q_tau = Af["dmft_loop/i_000/Q_tau"]

invg0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
G0 = invg0.copy()
G0.data[:,0,0] = g0.data[:,0,0]
# Initialize G0_iw (CT-INT uses G0_iw, not Delta_tau)
# Both spin channels get the same G0 (paramagnetic solution)
for bl, g_bl in S.G0_iw:
    g_bl.data[:,0,0] = G0.data[:,0,0]

Q_tau = GfImTime(target_shape=[1,1], statistic='Boson', beta=beta, n_points=n_tau_bosonic)
Q_iw = GfImFreq(target_shape=[1,1], statistic='Boson', beta=beta, n_points=n_tau_bosonic//2)
Q_tau.data[:,0,0] = q_tau.data[:,0,0]

#Q_iw = make_gf_from_fourier(Q_tau)
Q_iw << Fourier(Q_tau)
print(Q_tau)
print(Q_iw)
print(S.Jperp_iw)
# --------- Spin-spin interaction via Matsubara frequency (DLR) ---------
S.Jperp_iw.data[:]          = -1.00*J* Q_iw.data[:] * i_1
S.D0_iw["up", "up"].data[:] = -0.25*J* Q_iw.data[:] * i_2
S.D0_iw["dn", "dn"].data[:] = -0.25*J* Q_iw.data[:] * i_3
S.D0_iw["up", "dn"].data[:] =  0.25*J* Q_iw.data[:] * i_4
S.D0_iw["dn", "up"].data[:] =  0.25*J* Q_iw.data[:] * i_5


S.solve(h_int=h_int,
        n_cycles = 1000000,
        length_cycle = 100,
        n_warmup_cycles = 100,
        random_seed = 34788,
        measure_histogram = True,
        measure_density = True,
        measure_M4_iw = True,
        n_iw_M4 = 5,
        nfft_buf_size = 50,
        measure_M3pp_tau = True,
        measure_M3ph_tau = True,
        measure_M3xph_tau = True,
        n_iw_M3 = 10,
        n_iW_M3 = 10,
        n_tau_M3 = 41,
        measure_chi2pp_tau = True,
        measure_chi2ph_tau = True,
        n_iw_chi2 = 10,
        n_tau_chi2 = 21,
        measure_chiAB_tau = True,
        chi_A_vec = [n('up',0) + n('dn', 0)],
        chi_B_vec = [n('up',0) + n('dn', 0)],
        post_process = True )
# -------- Save in Aive ---------
if mpi.is_master_node():
    filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_ctint_old_J-{J}-U-{U}_{i_1}_{i_2}_{i_3}_{i_4}_{i_5}_b-{beta}.h5"
    with h5.HDFArchive(filename, "w") as A:
        A["G0_iw"] = S.G0_iw
        A["G_iw"] = S.G_iw
        A["G2_iw"] = S.G2_iw
        A["chi3pp_iw"] = S.chi3pp_iw
        A["chi3ph_iw"] = S.chi3ph_iw
        A["chi3xph_iw"] = S.chi3xph_iw
        A["chi2pp_iw"] = S.chi2pp_iw
        A["chi2ph_iw"] = S.chi2ph_iw
        A["chiAB_iw"] = S.chiAB_iw
        A["chi2pp_tau_from_M3"] = S.chi2pp_tau_from_M3
        A["chi2ph_tau_from_M3"] = S.chi2ph_tau_from_M3
        A["chi2xph_tau_from_M3"] = S.chi2xph_tau_from_M3

