"""
Single orbital with dynamical spin-spin interactions, using the updated DLR setup.
The script keeps only the CT-INT observables needed for this benchmark: G_iw and chiAB_tau.
"""

import argparse

import h5
import triqs.utility.mpi as mpi
from triqs.gf import *
from triqs.gf.tools import *
from triqs.operators import n  # pyright: ignore[reportMissingImports]
from triqs_ctint import Solver


def parse_args():
    parser = argparse.ArgumentParser(description="Run spin-spin benchmarking (CT-INT).")
    parser.add_argument("--U", type=float, default=4.0, help="U parameter")
    parser.add_argument("--J", type=float, default=1.0, help="J parameter")
    parser.add_argument("--i1", type=float, default=1.0, help="i1 switch (0 or 1)")
    parser.add_argument("--i2", type=float, default=1.0, help="i2 switch (0 or 1)")
    parser.add_argument("--i3", type=float, default=1.0, help="i3 switch (0 or 1)")
    parser.add_argument("--i4", type=float, default=1.0, help="i4 switch (0 or 1)")
    parser.add_argument("--i5", type=float, default=1.0, help="i5 switch (0 or 1)")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    return parser.parse_args()


args = parse_args()

beta = args.beta
U = args.U
J = args.J
i_1, i_2, i_3, i_4, i_5 = args.i1, args.i2, args.i3, args.i4, args.i5
print(U,J,i_1,i_2,i_3,i_4,i_5,beta)
n_tau = 2001
n_tau_bosonic = 2001
dlr_wmax = 10.0
dlr_eps = 1e-10

gf_struct = [("down", 1), ("up", 1)]
Sz = 0.5 * (n("up", 0) - n("down", 0))

solver = Solver(
    beta=beta,
    gf_struct=gf_struct,
    n_tau=n_tau,
    use_Jperp=True,
    use_D=True,
    dlr_wmax=dlr_wmax,
)

with h5.HDFArchive("ctint.ref.h5", "r") as archive:
    g0 = archive["dmft_loop/i_001/S/G0_iw/up"]
    q_tau_ref = archive["dmft_loop/i_000/Q_tau"]

g0_tau = make_gf_from_fourier(g0)
g0_tau_dlr = fit_gf_dlr(g0_tau, w_max=dlr_wmax, eps=dlr_eps, symmetrize=True)
g0_iw = make_gf_dlr_imfreq(g0_tau_dlr)
for _, g0_block in solver.G0_iw:
    g0_block.data[:, 0, 0] = g0_iw.data[:, 0, 0]

q_tau = GfImTime(target_shape=[1, 1], statistic="Boson", beta=beta, n_points=n_tau_bosonic)
q_tau <<= q_tau_ref
q_tau_dlr = fit_gf_dlr(q_tau, w_max=dlr_wmax, eps=dlr_eps, symmetrize=True)
q_iw_dlr = make_gf_dlr_imfreq(q_tau_dlr)

solver.Jperp_iw.data[:] = -(J/2) * q_iw_dlr.data[:] * i_1
solver.D0_iw["up", "up"].data[:] = -0.125 * J * q_iw_dlr.data[:] * i_2
solver.D0_iw["down", "down"].data[:] = -0.125 * J * q_iw_dlr.data[:] * i_3
solver.D0_iw["up", "down"].data[:] = 0.125 * J * q_iw_dlr.data[:] * i_4
solver.D0_iw["down", "up"].data[:] = 0.125 * J * q_iw_dlr.data[:] * i_5

solver.solve(
    h_int=U * n("up", 0) * n("down", 0),
    n_cycles=5000000,
    measure_M_iw=True,
    measure_M_tau=False,
    measure_chiAB_tau=True,
    chi_ops=[(Sz, Sz)],
    post_process=True,
)

if mpi.is_master_node():
    filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_ctint_J-{J}-U-{U}_{i_1}_{i_2}_{i_3}_{i_4}_{i_5}_b-{beta}.h5"
    with h5.HDFArchive(filename, "w") as archive:
        archive["G0_iw"] = solver.G0_iw
        archive["G_iw"] = solver.G_iw
        archive["chiAB_tau"] = solver.chiAB_tau