from triqs.gfs import *
import numpy as np
import triqs.utility.mpi as mpi
from triqs.operators import n
from triqs_ctseg import Solver as SolverSeg
from triqs_cthyb import Solver as SolverHyb

L = 0.1
w0 = 1.0
beta = 20.0
U = 4.0
hopping = 1.0

n_tau = 1000
n_tau_bosonic = 1000
n_iw = 500

gf_struct = [('down', 1), ('up', 1)]
q_tau = GfImTime(indices=[0], statistic='Boson', beta=beta, n_points=n_tau_bosonic)
q_iw = make_gf_from_fourier(q_tau)
q_iw << Function(lambda w: 2 * L/w0 * w0**2 / (w**2 - w0**2))
q_tau << Fourier(q_iw)

Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[q_tau, q_tau], [q_tau, q_tau]])
Q_iw = make_gf_from_fourier(Q_tau)
ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
zero_freq = np.where(np.abs(ivn) < 1e-10)
q_iw_0 = np.real((Q_iw["up", "up"].data[zero_freq][0,0,0]+Q_iw["up", "down"].data[zero_freq][0,0,0]))/2.0

mu = U/2 + q_iw_0

# Run CTSEG
S_seg = SolverSeg(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)
S_seg.Delta_tau << Fourier(SemiCircular(2*hopping))
S_seg.D0_tau << Q_tau
solve_params_seg = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 1000,
    "n_cycles": 10000,
    "measure_nn_static": True,
}
S_seg.solve(**solve_params_seg)
if mpi.is_master_node():
    print(f"CTSEG Densities: {S_seg.results.densities}")
    print(f"CTSEG nn_static: {S_seg.results.nn_static}")

# Run CTHYB
S_hyb = SolverHyb(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic, delta_interface=True)
S_hyb.Delta_tau << Fourier(SemiCircular(2*hopping))
S_hyb.D0_tau << Q_tau
solve_params_hyb = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 1000,
    "n_cycles": 10000,
    "lang_firsov": True,
    "dyn_n_l": 100,
    "measure_density_matrix": True,
    "use_norm_as_weight": True,
}
S_hyb.solve(**solve_params_hyb)
if mpi.is_master_node():
    from triqs.atom_diag import trace_rho_op
    rho = S_hyb.density_matrix
    h_loc_diag = S_hyb.h_loc_diagonalization
    n_up_val = trace_rho_op(rho, n("up", 0), h_loc_diag).real
    n_dn_val = trace_rho_op(rho, n("down", 0), h_loc_diag).real
    docc = trace_rho_op(rho, n("up", 0) * n("down", 0), h_loc_diag).real
    print(f"CTHYB Densities: up={n_up_val}, dn={n_dn_val}")
    print(f"CTHYB docc: {docc}")
