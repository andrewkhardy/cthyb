import triqs.utility.mpi as mpi
from triqs.gfs import *
from triqs.operators import *
from h5 import HDFArchive
from triqs.utility.comparison_tests import *

from triqs_cthyb import *

#  Example of DMFT single site solution with CTQMC

# set up a few parameters
half_bandwidth = 1.0
U = 2.5
mu = (U/2.0)+0.2
beta = 80.0

gf_struct = [['down',1], ['up',1]]

# Parameters
p = {}
p["max_time"] = -1
p["random_name"] = ""
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 1000
p["n_warmup_cycles"] = 5000
p["n_cycles"] = 20000
p["measure_G_l"] = True
p["move_double"] = False
p["perform_tail_fit"] = True
p["fit_max_moment"] = 3
p["fit_min_w"] = 1.2
p["fit_max_w"] = 3.0

# Construct solver (delta interface: provide Delta_tau instead of G0_iw)
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=8001, n_l=30, delta_interface=True)

# Local Hamiltonian
H = U*n("up",0)*n("down",0)

# Single-particle part of local Hamiltonian (chemical potential)
h_loc0 = -mu * (n("up",0) + n("down",0))

# init the Green function
S.G_iw << SemiCircular(half_bandwidth)

t = half_bandwidth / 2.0

for i in range(2):

    g = 0.5 * ( S.G_iw['up'] + S.G_iw['down'] )
    # Bethe lattice self-consistency: Delta(iw) = t^2 * G(iw)
    Delta_iw = S.G_iw.copy()
    for name, d in Delta_iw:
        d << t**2 * g
    S.Delta_tau << Fourier(Delta_iw)

    S.solve(h_int=H, h_loc0=h_loc0, **p)

# Calculation is done. Now save a few things
if mpi.is_master_node():
    with HDFArchive("single_site_bethe.out.h5",'w') as Results:

        Results["Delta_tau"] = S.Delta_tau

        Results["G_tau"] = S.G_tau
        Results["G_l"] = S.G_l

        Results["G_iw"] = S.G_iw
        Results["G_iw_raw"] = S.G_iw_raw

        Results["Sigma_iw"] = S.Sigma_iw
        Results["Sigma_iw_raw"] = S.Sigma_iw_raw

# Check against reference
if mpi.is_master_node():
    with HDFArchive("single_site_bethe.ref.h5",'r') as Results:

        assert_block_gfs_are_close(Results["G_tau"], S.G_tau)
        assert_block_gfs_are_close(Results["G_l"], S.G_l)

        assert_block_gfs_are_close(Results["G_iw"], S.G_iw)
        assert_block_gfs_are_close(Results["G_iw_raw"], S.G_iw_raw)

        assert_block_gfs_are_close(Results["Sigma_iw"], S.Sigma_iw, precision=1e-4)

        assert_block_gfs_are_close(Results["Delta_tau"], S.Delta_tau)

