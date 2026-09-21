# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Exact-diagonalization reference for the two-patch dimer with a retarded spin-spin
interaction (model.py, run with --bath discrete so the model is ED-representable).

    H = sum_{K s} (eps_K - mu) n_{K s}                       patch levels
      + U sum_i n_{i up} n_{i down}                          Hubbard, local on the SITES
      + sum_{K s} [eps_bath b^dag b + V (c^dag b + h.c.)]    one bath site per patch/spin
      + sum_{m a} omega_0 d^dag_{m a} d_{m a}
      + sum_{m a} g_m T_m^a (d_{m a} + d^dag_{m a}) / sqrt(2 omega_0)

Integrating out the boson triplets reproduces the retarded interaction the QMC samples,
`sum_ij lambda_ij(tau) S_i(tau).S_j(0)` with `lambda_ij = -J_ij Q(tau)`.

Which J are representable, and why that is a real constraint
------------------------------------------------------------
Integrating out harmonic bosons coupled linearly to operators {T_m} always gives
`sum_m g_m^2 Q_m(tau) T_m(tau) T_m(0)`. Since `g_m^2 >= 0` and Q has a fixed sign, the
reachable set of kernels is exactly `-(positive semidefinite) x |Q|`. So a boson-mode ED
of this model exists **iff -J is positive semidefinite**, i.e. `J_intra <= -|J_inter|`.

That is not a technicality to work around: the physically interesting DCA point
`J_intra = 0, J_inter = 0.5` has eigenvalues +-0.5 and is therefore **not** representable
by any real-boson Hamiltonian. CTHYB samples it perfectly well -- it is a well-defined
action -- but there is nothing for ED to diagonalize. This script raises rather than
silently building a non-Hermitian H, and names the offending eigenvalues.

The nearest representable point is `J_intra = -J_inter`, where -J has rank 1 and the
single channel is `u = (1, -1)/sqrt(2)`, i.e. one boson triplet coupled to
`(S_1 - S_2)/sqrt(2)`. That is exactly how an antiferromagnetic `S_1.S_2` arises
physically: `-g^2 (S_1 - S_2)^2` contains `+2 g^2 S_1.S_2` plus on-site `S_i^2` terms.

Cost
----
8 fermion modes (4 impurity + 4 bath) = 256 states, blocked by total N (conserved: the
boson coupling breaks S_z but not charge), so the largest block is C(8,4) = 70. The boson
space is `(n_ph+1)^(3*rank)`. Rank 1 with n_ph = 3 gives 70*64 = 4480 -- a minute or two.
Rank 2 needs `(n_ph+1)^6` and gets expensive fast; if it is ever needed, the route is to
block additionally by `J_z = S_z + sum_m (n_{m,+1} - n_{m,-1})`, conserved for an
SU(2)-covariant S.B coupling in a spherical boson basis. Not implemented.

H is complex Hermitian because S^y is imaginary; `eigh` handles that directly.
"""
import argparse
import os
import time

import numpy as np
from h5 import HDFArchive

import model as model_def

parser = argparse.ArgumentParser(description='ED reference: two-patch dimer + discrete bath + spin bosons.')
model_def.add_model_args(parser)
parser.add_argument('--n_ph', type=int, default=3, help='Phonon levels kept per boson mode, minus one')
parser.add_argument('--n_ph_check', type=int, default=1,
                    help='Also solve with n_ph + this many levels and report the difference (0 to skip)')
parser.add_argument('--n_tau', type=int, default=201, help='Imaginary-time points on [0, beta]')
parser.add_argument('--n_iw', type=int, default=256, help='Positive Matsubara frequencies')
parser.add_argument('--psd_tol', type=float, default=1e-10,
                    help='Eigenvalues of -J below -psd_tol make the model non-representable')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()

if args.bath != 'discrete':
    raise SystemExit("run_ed.py needs --bath discrete: the 'dca' bath is a continuous "
                     "coarse-grained hybridization and has no finite Hamiltonian representation.")
M = model_def.Model(args)

N_PATCH = model_def.N_PATCH
SPINS = model_def.SPIN_NAMES
labels = [(s, K) for s in SPINS for K in range(N_PATCH)]   # impurity spin-orbitals
n_so = len(labels)                                          # 4
n_modes = 2 * n_so                                          # + one bath site each = 8
dim_f = 2 ** n_modes


# --------------------------------------------------------------------------- fermions
def annihilators(n):
    """Jordan-Wigner annihilation operators; mode 0 is the leftmost tensor factor."""
    lower, Z, I = np.array([[0., 1.], [0., 0.]]), np.diag([1., -1.]), np.eye(2)
    ops = []
    for j in range(n):
        m = np.array([[1.]])
        for k in range(n):
            m = np.kron(m, Z if k < j else (lower if k == j else I))
        ops.append(m)
    return ops


c = annihilators(n_modes)
occ = np.array([[(s >> (n_modes - 1 - j)) & 1 for j in range(n_modes)] for s in range(dim_f)])
mode_of = {lab: a for a, lab in enumerate(labels)}


def c_patch(spin, K):
    return c[mode_of[(spin, K)]]


# Site operators are rotations of the patch operators: c_i = sum_K R[i,K] c_K.
R = M.R
c_site = {(i, s): sum(R[i, K] * c_patch(s, K) for K in range(N_PATCH))
          for i in range(N_PATCH) for s in SPINS}

# Spin operators on each site, as matrices on the fermion Fock space.
Sx, Sy, Sz = [], [], []
for i in range(N_PATCH):
    up, dn = c_site[(i, 'up')], c_site[(i, 'down')]
    sp = up.T @ dn            # S^+ = c^dag_up c_down
    sm = dn.T @ up            # S^-
    Sx.append(0.5 * (sp + sm))
    Sy.append(-0.5j * (sp - sm))
    Sz.append(0.5 * (up.T @ up - dn.T @ dn))

# --------------------------------------------------------------- boson decomposition
J = np.array([[M.J_intra, M.J_inter], [M.J_inter, M.J_intra]])
evals, evecs = np.linalg.eigh(-J)
if np.any(evals < -args.psd_tol):
    raise SystemExit(
        f"-J is not positive semidefinite (eigenvalues {np.round(evals, 6)}), so this model has NO\n"
        f"real-boson Hamiltonian and cannot be diagonalized. Integrating out harmonic bosons always\n"
        f"gives a kernel -(psd) x |Q|, which requires J_intra <= -|J_inter|; here J_intra="
        f"{M.J_intra}, J_inter={M.J_inter}.\n"
        f"CTHYB can still sample this point (it is a well-defined action) -- run it with\n"
        f"run_cthyb.py and use the lang_firsov True/False pair as the cross-check instead.\n"
        f"The nearest ED-representable point is J_intra = -J_inter = {-abs(M.J_inter)}.")

channels = [(float(np.sqrt(max(ev, 0.0))), evecs[:, k])
            for k, ev in enumerate(evals) if ev > args.psd_tol]
rank = len(channels)
n_bos = 3 * rank
print(f"-J eigenvalues {np.round(evals, 6)} -> {rank} boson channel(s), {n_bos} modes")
for g_m, u in channels:
    print(f"    g = {g_m:.6f}  coupling to  T = {np.round(u, 6)} . S")

# T_m^alpha = sum_i u_m[i] S_i^alpha, one per channel and Cartesian component.
T_ops = []
for g_m, u in channels:
    for S_alpha in (Sx, Sy, Sz):
        T_ops.append((g_m, sum(u[i] * S_alpha[i] for i in range(N_PATCH))))


# ------------------------------------------------------------------- fermion Hamiltonian
H_f = np.zeros((dim_f, dim_f), dtype=complex)
for s in SPINS:
    for K in range(N_PATCH):
        op = c_patch(s, K)
        H_f += (M.eps_patch[K] - M.mu) * (op.T @ op)
for i in range(N_PATCH):
    nup = c_site[(i, 'up')].T @ c_site[(i, 'up')]
    ndn = c_site[(i, 'down')].T @ c_site[(i, 'down')]
    H_f += M.U * (nup @ ndn)
for a, lab in enumerate(labels):                      # bath: one site per spin-orbital
    b = c[n_so + a]
    H_f += M.eps_bath * (b.T @ b) + M.V * (c[a].T @ b + b.T @ c[a])

# Total charge is conserved by every term (the S.B coupling moves spin, not charge).
N_tot = occ.sum(axis=1)
block_keys = sorted(set(N_tot))

tau = np.linspace(0.0, M.beta, args.n_tau)
w_n = (2 * np.arange(args.n_iw) + 1) * np.pi / M.beta
iw = 1j * w_n


def boson_operators(n_lev):
    """(H_boson, [x_m]) on the tensor product of `n_bos` ladders truncated at `n_lev`."""
    ph = np.arange(n_lev)
    d = np.diag(np.sqrt(ph[1:]), 1)
    x1 = (d + d.T) / np.sqrt(2 * M.omega_0)
    num = np.diag(ph).astype(float)
    eye = np.eye(n_lev)

    def embed(mat, slot):
        out = np.array([[1.0]])
        for k in range(n_bos):
            out = np.kron(out, mat if k == slot else eye)
        return out

    dim_b = n_lev ** n_bos
    H_b = np.zeros((dim_b, dim_b))
    xs = []
    for slot in range(n_bos):
        H_b += M.omega_0 * embed(num, slot)
        xs.append(embed(x1, slot))
    return H_b, xs


def solve(n_lev):
    """Diagonalize block by block; return correlators and phonon diagnostics."""
    H_b, xs = boson_operators(n_lev)
    dim_b = n_lev ** n_bos
    blocks = {}
    for key in block_keys:
        idx = np.where(N_tot == key)[0]
        H = np.kron(H_f[np.ix_(idx, idx)], np.eye(dim_b)) + np.kron(np.eye(len(idx)), H_b)
        for (g_m, T), x in zip(T_ops, xs):
            H += g_m * np.kron(T[np.ix_(idx, idx)], x)
        E, U = np.linalg.eigh(H)
        blocks[key] = (idx, E, U, dim_b)
    return correlators(blocks, n_lev)


def correlators(blocks, n_lev):
    E0 = min(E.min() for _, E, _, _ in blocks.values())
    Z = sum(np.exp(-M.beta * (E - E0)).sum() for _, E, _, _ in blocks.values())

    G = np.zeros((n_so, len(tau)))
    G_iw = np.zeros((n_so, len(iw)), dtype=complex)
    chi_zz = np.zeros((N_PATCH, N_PATCH, len(tau)))
    mean_phonons = 0.0

    ph_number = np.diag(np.add.reduce(
        [np.kron(np.kron(np.eye(n_lev ** s), np.diag(np.arange(n_lev))),
                 np.eye(n_lev ** (n_bos - s - 1))) for s in range(n_bos)]))

    for key, (idx, E, U, dim_b) in blocks.items():
        wl = np.exp(-np.outer(M.beta - tau, E - E0))
        wr = np.exp(-np.outer(tau, E - E0))
        boltz = np.exp(-M.beta * (E - E0))

        # <S_i^z(tau) S_j^z(0)>: S^z is diagonal in the boson factor.
        sz = [U.conj().T @ np.kron(Sz[i][np.ix_(idx, idx)], np.eye(dim_b)) @ U for i in range(N_PATCH)]
        for i in range(N_PATCH):
            for j in range(N_PATCH):
                chi_zz[i, j] += ((wl @ (sz[i] * sz[j].conj()).real) * wr).sum(axis=1)

        # G_a(tau) and G_a(iw), m one particle fewer than n
        if key - 1 in blocks:
            idx_m, E_m, U_m, _ = blocks[key - 1]
            for a in range(n_so):
                C = U_m.conj().T @ np.kron(c[a][np.ix_(idx_m, idx)], np.eye(dim_b)) @ U
                C2 = np.abs(C) ** 2
                G[a] -= ((np.exp(-np.outer(M.beta - tau, E_m - E0)) @ C2) * wr).sum(axis=1)
                dE = E_m[:, None] - E[None, :]
                w8 = np.exp(-M.beta * (E_m - E0))[:, None] + np.exp(-M.beta * (E - E0))[None, :]
                G_iw[a] += ((C2 * w8)[None, :, :] / (iw[:, None, None] + dE[None, :, :])).sum(axis=(1, 2))

        mean_phonons += boltz @ (np.tile(ph_number, len(idx)) @ (np.abs(U) ** 2))

    return G / Z, G_iw / Z, chi_zz / Z, mean_phonons / Z


start = time.time()
largest = max(len(np.where(N_tot == k)[0]) for k in block_keys) * (args.n_ph + 1) ** n_bos
print(f"largest block {largest} ({n_bos} boson modes x {args.n_ph + 1} levels)")
G, G_iw, chi_zz, mean_phonons = solve(args.n_ph + 1)
print(f"ED done in {time.time() - start:.1f} s, <N_ph> = {mean_phonons:.4f}")

truncation = float('nan')
if args.n_ph_check > 0:
    G_c, _, chi_c, _ = solve(args.n_ph + 1 + args.n_ph_check)
    truncation = max(np.abs(G - G_c).max(), np.abs(chi_zz - chi_c).max())
    print(f"Phonon truncation: max|X(n_ph) - X(n_ph + {args.n_ph_check})| = {truncation:.2e}")

occupations = -G[:, -1]
print("<n_a> =", np.round(occupations, 5))
print(f"max |G_a(0) + G_a(beta) + 1| = {np.abs(G[:, 0] + G[:, -1] + 1).max():.2e}")

# Sigma, from the same inputs the QMC side is given.
delta_iw = M.V ** 2 / (iw - M.eps_bath)
Sigma_iw = np.zeros_like(G_iw)
for a, (s, K) in enumerate(labels):
    Sigma_iw[a] = (iw + M.mu - M.eps_patch[K] - delta_iw) - 1.0 / G_iw[a]
print(f"Sigma: Im<0 on the first 20 w_n for every orbital: {bool(np.all(Sigma_iw[:, :20].imag < 0))}")

# S_tot.S_tot is basis independent, so for uniform J the total-spin channel must agree
# between the site and patch bases -- the same identity check_rotation.py uses on the
# vertex expander, here applied to the solved correlator.
chi_tot = chi_zz.sum(axis=(0, 1))
print(f"sum_ij chi^zz_ij(0) = {chi_tot[0]:.6f}   (= <(S^z_tot)^2>)")

os.makedirs(args.out_dir, exist_ok=True)
filename = os.path.join(args.out_dir, f"ed_{M.tag()}_nph-{args.n_ph}.h5")
with HDFArchive(filename, 'w') as A:
    A['solver'] = 'ed'
    A['tau'] = tau
    A['G'] = G
    A['w_n'] = w_n
    A['G_iw'] = G_iw
    A['Sigma'] = Sigma_iw
    A['chi_zz'] = chi_zz
    A['chi_zz_total'] = chi_tot
    # Same keys CTHYB writes for its O_tau, so the plot compares them directly.
    A['tau_corr'] = tau
    A['corr'] = chi_tot
    A['labels'] = [f"{s},{K}" for s, K in labels]
    A['mu'] = M.mu
    A['density'] = occupations
    A['params'] = M.params()
    A['n_ph'] = args.n_ph
    A['ed_truncation'] = truncation
    A['mean_phonons'] = mean_phonons
    A['boson_channels'] = np.array([g for g, _ in channels])
print(f"Saved {filename}")
