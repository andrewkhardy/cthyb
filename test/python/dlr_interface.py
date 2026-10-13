"""
Test of the DLR mode of the Solver (Solver(..., dlr_wmax=...)): G0_iw or Delta_iw in, G_iw and Sigma_iw out, all on
the DLR Matsubara mesh.

  1. G0_iw input: h_loc0 and Delta_iw are recovered from G0_iw, and Delta_tau is the same as with the Delta_iw input
  2. U = 0: G_iw is G0_iw within the Monte Carlo error and Sigma_iw vanishes at low frequency
  3. U > 0: G_iw agrees with the DLR fit of the regular mode's G_tau; dlr_sigma='crm' gives Sigma_iw on the DLR mesh
  4. add_dyn_int takes D on a DLR mesh
"""

import numpy as np

import triqs.utility.mpi as mpi
from triqs.gfs import BlockGf, Gf, MeshImFreq, MeshDLR, MeshDLRImFreq, iOmega_n, inverse
from triqs.gfs import make_gf_imtime, make_gf_dlr_imfreq, fit_gf_dlr
from triqs.operators import n, c, c_dag
from triqs.operators.util import block_matrix_from_op

from triqs_cthyb import Solver

beta, dlr_wmax, dlr_eps = 10.0, 10.0, 1e-10
seed = 123 * mpi.rank + 567


def hyb(mesh, E, V):
    """Delta(iw) = V [1/(iw - E) + 1/(iw + E)] V on any Matsubara mesh."""
    g = Gf(mesh=mesh, target_shape=E.shape)
    g << inverse(iOmega_n - E) + inverse(iOmega_n + E)
    g.from_L_G_R(V, g, V)
    return g


def max_diff(a, b):
    return max(np.abs(ga.data - gb.data).max() for (_, ga), (_, gb) in zip(a, b))


dlr_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=dlr_wmax, eps=dlr_eps, symmetrize=True)


# ---------- 1. G0_iw input: h_loc0, Delta_iw and Delta_tau ----------
mpi.report("=== DLR mode: h_loc0 and Delta from G0_iw ===")
gf_struct = [('0', 2)]
E_loc = np.array([[0.20, 0.30], [0.30, 0.40]])
Delta = BlockGf(name_list=['0'], block_list=[hyb(dlr_mesh, np.array([[1.00, 0.75], [0.75, -1.20]]),
                                                 np.array([[1.00, 0.25], [0.25, -1.00]]))])
h_int = n('0', 0) * n('0', 1)
short = dict(h_int=h_int, n_cycles=10, n_warmup_cycles=0, length_cycle=1, random_seed=seed, measure_G_tau=False)

S_g0 = Solver(beta=beta, gf_struct=gf_struct, dlr_wmax=dlr_wmax, dlr_eps=dlr_eps)
S_g0.G0_iw['0'] << inverse(iOmega_n - E_loc - Delta['0'])
S_g0.solve(**short)

h_loc0 = block_matrix_from_op(S_g0.h_loc0, gf_struct)[0]
assert np.abs(h_loc0.imag).max() == 0.0, f"h_loc0 should be real: {h_loc0}"
assert np.abs(h_loc0 - E_loc).max() < 1e-5, f"h_loc0 from G0_iw: {h_loc0}, expected {E_loc}"
assert max_diff(S_g0.Delta_iw, Delta) < 1e-5, "Delta_iw from G0_iw"

S_d = Solver(beta=beta, gf_struct=gf_struct, delta_interface=True, dlr_wmax=dlr_wmax, dlr_eps=dlr_eps)
S_d.Delta_iw << Delta
h_loc0_op = sum(c_dag('0', i) * E_loc[i, j] * c('0', j) for i in range(2) for j in range(2))
S_d.solve(h_loc0=h_loc0_op, **short)

assert max_diff(S_d.Delta_tau, make_gf_imtime(Delta, S_d.n_tau)) < 1e-10, "Delta_tau from Delta_iw"
assert max_diff(S_d.Delta_tau, S_g0.Delta_tau) < 1e-5, "Delta_tau from G0_iw and from Delta_iw"

try:
    S_g0.solve(h_loc0=h_loc0_op, **short)
    raise AssertionError("h_loc0 with the G0_iw input should raise")
except ValueError:
    pass


# ---------- Sign-free single-orbital model for the Monte Carlo checks ----------
gf_struct = [('up', 1), ('dn', 1)]
E_bath, V_bath = np.array([[1.0]]), np.array([[0.5]])
# The statistical error of G(iw) at the lowest frequencies, where |G| ~ 2, is about 0.02 with these settings (the
# regular mode has the same); the tolerances below are 2-3 times that
mc = dict(n_cycles=100000, n_warmup_cycles=1000, length_cycle=20, random_seed=seed)


def G0_of(mesh, eps):
    g0 = hyb(mesh, E_bath, V_bath)
    g0 << inverse(iOmega_n - eps - g0)
    return BlockGf(name_list=['up', 'dn'], block_list=[g0, g0])


def dlr_solver(eps):
    S = Solver(beta=beta, gf_struct=gf_struct, dlr_wmax=dlr_wmax, dlr_eps=dlr_eps)
    S.G0_iw << G0_of(dlr_mesh, eps)
    return S


# ---------- 2. U = 0 ----------
mpi.report("=== DLR mode: U = 0 ===")
S = dlr_solver(eps=0.2)
S.solve(h_int=0 * n('up', 0) * n('dn', 0), **mc)

assert S.G_iw.mesh == dlr_mesh and S.Sigma_iw.mesh == dlr_mesh
for bl, g in S.G_dlr:  # the fit holds the 1/iw moment: G(0) + G(beta) = -1
    ends = make_gf_imtime(g, 2).data[:, 0, 0]
    assert abs(ends[0] + ends[-1] + 1) < 1e-8, f"G_dlr[{bl}](0) + G_dlr[{bl}](beta) = {ends[0] + ends[-1]}"
assert max_diff(S.G_iw, S.G0_iw) < 0.05, f"G_iw != G0_iw at U = 0: {max_diff(S.G_iw, S.G0_iw)}"
low = [i for i, w in enumerate(dlr_mesh.values()) if abs((2 * w.n + 1) * np.pi / beta) < 2.0]
for bl, sigma in S.Sigma_iw:
    assert np.abs(sigma.data[low]).max() < 0.2, f"Sigma_iw[{bl}] != 0 at U = 0: {np.abs(sigma.data[low]).max()}"


# ---------- 3. U > 0: DLR mode against the regular mode, and CRM ----------
mpi.report("=== DLR mode: U = 2 against the regular mode, CRM ===")
U = 2.0
h_int = U * n('up', 0) * n('dn', 0)

S_reg = Solver(beta=beta, gf_struct=gf_struct)
S_reg.G0_iw << G0_of(S_reg.G0_iw.mesh, -U / 2)
S_reg.solve(h_int=h_int, **mc)

S = dlr_solver(-U / 2)
S.solve(h_int=h_int, measure_density_matrix=True, use_norm_as_weight=True, dlr_sigma='crm', **mc)

G_reg = make_gf_dlr_imfreq(fit_gf_dlr(S_reg.G_tau, MeshDLR(dlr_mesh)))
assert max_diff(S.G_iw, G_reg) < 0.08, f"DLR mode G_iw != regular mode: {max_diff(S.G_iw, G_reg)}"

Sigma_dyson = S.G0_iw.copy()
Sigma_dyson << inverse(S.G0_iw) - inverse(S.G_iw)
for bl, sigma in S.Sigma_iw:
    assert sigma.mesh == dlr_mesh and np.all(np.isfinite(sigma.data))
    diff = np.abs(sigma.data[low] - Sigma_dyson[bl].data[low]).max()
    assert diff < 0.2, f"CRM and Dyson Sigma_iw[{bl}] differ at low frequency by {diff}"


# ---------- 4. add_dyn_int with D on a DLR mesh ----------
mpi.report("=== DLR mode: add_dyn_int with a DLR coupling ===")
D_dlr = Gf(mesh=MeshDLRImFreq(beta=beta, statistic='Boson', w_max=dlr_wmax, eps=dlr_eps), target_shape=[])
D_dlr << 0.3 * (inverse(iOmega_n - 1.5) - inverse(iOmega_n + 1.5))
X = n('up', 0) + n('dn', 0)

G_tau = []
for D in (D_dlr, make_gf_imtime(D_dlr, 10001)):
    S = dlr_solver(-U / 2)
    S.add_dyn_int(D, X, X)
    S.solve(h_int=h_int, n_cycles=200, n_warmup_cycles=10, length_cycle=10, random_seed=seed)
    G_tau.append(S.G_tau.copy())
assert max_diff(*G_tau) < 1e-12, "add_dyn_int: DLR and imtime couplings give different runs"
