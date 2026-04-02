"""
Compare CT-INT and CT-SEG for a single-orbital model with dynamical spin-spin interactions.

Convention difference:
  CT-SEG action: (1/2) * Jperp * s+s- + (1/2) * D0 * n*n  (explicit 1/2 prefactor)
  CT-INT action: Jperp * s+s-  + D0 * n*n  (no 1/2 prefactor, two vertex types S+S-/S-S+)
  => Jperp_ctint = Jperp_ctseg / 2,  D0_ctint = D0_ctseg / 2
"""
from triqs_ctint import Solver as CtintSolver
from triqs_ctseg import SolverCore as CtsegSolver
from triqs.gf import *
from triqs.gf.tools import *
from triqs.operators import n
import triqs.utility.mpi as mpi
import h5
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ Physical parameters ============
beta = 10.0
U = 4.0
mu = U / 2
J = 1.0
n_tau = 2001
n_tau_bosonic = 2001
n_iw = 1025
n_cycles = 10000000

gf_struct = [('down', 1), ('up', 1)]
Sz = 0.5 * (n('up', 0) - n('down', 0))

# ============ Load reference data ============
with h5.HDFArchive("ctint.ref.h5", 'r') as Af:
    g0 = Af["dmft_loop/i_001/S/G0_iw/up"]
    q_tau = Af["dmft_loop/i_000/Q_tau"]

Q_tau = GfImTime(target_shape=[1, 1], statistic='Boson', beta=beta, n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = q_tau.data[:, 0, 0]

# ============ Run CT-SEG ============
print("=" * 60)
print("Running CT-SEG")
print("=" * 60)

S_seg = CtsegSolver(gf_struct=gf_struct, beta=beta, n_tau=4096, n_tau_bosonic=n_tau_bosonic)

# Hybridization
invg0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
invg0 << inverse(g0)
Delta = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
Delta << iOmega_n + mu - invg0
S_seg.Delta_tau << Fourier(Delta)

# Spin-spin interaction (CT-SEG convention)
S_seg.Jperp_tau << -(J) * Q_tau
S_seg.D0_tau["up", "up"] << -0.25 * J * Q_tau
S_seg.D0_tau["down", "down"] << -0.25 * J * Q_tau
S_seg.D0_tau["up", "down"] << 0.25 * J * Q_tau
S_seg.D0_tau["down", "up"] << 0.25 * J * Q_tau

S_seg.solve(
    h_int=U * n("up", 0) * n("down", 0),
    h_loc0=-mu * (n("up", 0) + n("down", 0)),
    length_cycle=100,
    n_warmup_cycles=100000,
    n_cycles=n_cycles,
    measure_nn_tau=True,
)

# ============ Run CT-INT ============
print("=" * 60)
print("Running CT-INT")
print("=" * 60)

S_int = CtintSolver(beta=beta, gf_struct=gf_struct, n_tau=n_tau,
                    use_Jperp=True, use_D=True, dlr_wmax=10.0)

# G0_iw
g0_tau = make_gf_from_fourier(g0)
g0_tau_dlr = fit_gf_dlr(g0_tau, w_max=10.0, eps=1e-10, symmetrize=True)
g0_iw = make_gf_dlr_imfreq(g0_tau_dlr)
for bl, g_bl in S_int.G0_iw:
    g_bl.data[:, 0, 0] = g0_iw.data[:, 0, 0]

# Convert Q_tau to DLR
Q_tau_dlr = fit_gf_dlr(Q_tau, w_max=10.0, eps=1e-10, symmetrize=True)
Q_iw_dlr = make_gf_dlr_imfreq(Q_tau_dlr)

# Spin-spin interaction (CT-INT convention = CT-SEG / 2)
S_int.Jperp_iw.data[:] = -(J / 2) * Q_iw_dlr.data[:]
S_int.D0_iw["up", "up"].data[:] = -0.125 * J * Q_iw_dlr.data[:]
S_int.D0_iw["down", "down"].data[:] = -0.125 * J * Q_iw_dlr.data[:]
S_int.D0_iw["up", "down"].data[:] = 0.125 * J * Q_iw_dlr.data[:]
S_int.D0_iw["down", "up"].data[:] = 0.125 * J * Q_iw_dlr.data[:]

S_int.solve(
    h_int=U * n("up", 0) * n("down", 0),
    n_cycles=n_cycles,
    measure_M_iw=True,
    measure_M_tau=False,
    measure_chiAB_tau=True,
    chi_ops=[(Sz, Sz)],
    post_process=True,
)

if not mpi.is_master_node():
    exit()

# ============ Extract observables ============

# --- G(tau) ---
# CT-SEG: direct G_tau
G_seg_tau = S_seg.results.G_tau['up']
tau_seg = np.array([float(t) for t in G_seg_tau.mesh])
G_seg_data = G_seg_tau.data[:, 0, 0].real

# CT-INT: G_iw is on DLR mesh, convert to tau via DLR
G_int_dlr = make_gf_dlr(S_int.G_iw['up'])
G_int_tau = make_gf_imtime(G_int_dlr, len(tau_seg))
tau_int = np.array([float(t) for t in G_int_tau.mesh])
G_int_data = G_int_tau.data[:, 0, 0].real

# --- <Sz(tau) Sz(0)> ---
# CT-SEG: from nn_tau: <Sz(tau)Sz> = (1/4)[nn_uu + nn_dd - nn_ud - nn_du]
nn = S_seg.results.nn_tau
SzSz_seg = 0.25 * (nn['up', 'up'].data[:, 0, 0].real
                    + nn['down', 'down'].data[:, 0, 0].real
                    - nn['up', 'down'].data[:, 0, 0].real
                    - nn['down', 'up'].data[:, 0, 0].real)
tau_nn = np.array([float(t) for t in nn['up', 'up'].mesh])

# CT-INT: from chiAB_tau (DLR imtime mesh, tensor_valued<1> with pair index)
# chiAB_tau measures the full <A(tau)B(0)>. The DLR representation assumes a
# spectral (Lehmann) representation, which only holds for the connected part.
# Subtract the disconnected part <A><B> before DLR interpolation.
chiAB_raw = S_int.chiAB_tau.copy()
seg_dens = S_seg.results.densities
Sz_mean = 0.5 * (seg_dens['up'] - seg_dens['down'])
chiAB_raw.data[:, 0] -= (Sz_mean * Sz_mean).real
chiAB_conn_dlr = make_gf_dlr(chiAB_raw)
chiAB_conn_reg = make_gf_imtime(chiAB_conn_dlr, len(tau_nn))
# Add back the disconnected part for comparison with CT-SEG (which also gives the full correlator)
SzSz_int = chiAB_conn_reg.data[:, 0].real + (Sz_mean * Sz_mean).real
tau_chiAB = np.array([float(t) for t in chiAB_conn_reg.mesh])

# ============ Plots ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- G(tau) comparison ---
ax = axes[0, 0]
ax.plot(tau_seg, G_seg_data, '-', label='CT-SEG', linewidth=1.5)
ax.plot(tau_int, G_int_data, '--', label='CT-INT', linewidth=1.5)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$G(\tau)$')
ax.set_title(r'$G_\uparrow(\tau)$')
ax.legend()

# --- G(tau) difference ---
# Interpolate CT-INT onto CT-SEG grid for difference
from scipy.interpolate import interp1d
G_int_interp = interp1d(tau_int, G_int_data, kind='cubic')
tau_common = tau_seg[(tau_seg >= tau_int[0]) & (tau_seg <= tau_int[-1])]
G_seg_common = interp1d(tau_seg, G_seg_data, kind='cubic')(tau_common)
G_int_common = G_int_interp(tau_common)

ax = axes[0, 1]
ax.plot(tau_common, G_seg_common - G_int_common, '-', linewidth=1.5)
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\Delta G(\tau)$')
ax.set_title(r'$G_\uparrow^\mathrm{SEG}(\tau) - G_\uparrow^\mathrm{INT}(\tau)$')

# --- <Sz(tau)Sz(0)> comparison ---
ax = axes[1, 0]
ax.plot(tau_nn, SzSz_seg, '-', label='CT-SEG', linewidth=1.5)
ax.plot(tau_chiAB, SzSz_int, '--', label='CT-INT', linewidth=1.5)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\langle S_z(\tau) S_z(0) \rangle$')
ax.set_title(r'$\langle S_z(\tau) S_z(0) \rangle$')
ax.legend()

# --- <Sz(tau)Sz(0)> difference ---
SzSz_int_interp = interp1d(tau_chiAB, SzSz_int, kind='cubic')
tau_common_b = tau_nn[(tau_nn >= tau_chiAB[0]) & (tau_nn <= tau_chiAB[-1])]
SzSz_seg_common = interp1d(tau_nn, SzSz_seg, kind='cubic')(tau_common_b)
SzSz_int_common = SzSz_int_interp(tau_common_b)

ax = axes[1, 1]
ax.plot(tau_common_b, SzSz_seg_common - SzSz_int_common, '-', linewidth=1.5)
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\Delta \langle S_z S_z \rangle$')
ax.set_title(r'$\langle S_z S_z \rangle^\mathrm{SEG} - \langle S_z S_z \rangle^\mathrm{INT}$')

fig.suptitle(f'CT-INT vs CT-SEG comparison (U={U}, J={J}, beta={beta}, {n_cycles/1e6:.0f}M cycles)',
             fontsize=14)
fig.tight_layout()
fig.savefig('comparison_ctint_ctseg.pdf')
fig.savefig('comparison_ctint_ctseg.png', dpi=150)
print("\nPlots saved to comparison_ctint_ctseg.pdf / .png")
print("Done.")
