# Exact-diagonalization reference for the model in model.py: impurity + one bath site per
# spin-orbital + one phonon, solved directly as a Hamiltonian. Computes, exactly up to the
# phonon truncation,
#   chi_ab(tau) = <n_a(tau) n_b(0)>   for every pair of impurity spin-orbitals,
#   G_a(tau)    = -<c_a(tau) c_a^dag(0)>,
# to test CTHYB's estimators against (plot_kanamori_phonon.py). Nothing here uses Lang-Firsov,
# the kink estimator or any CTHYB code; only the Kanamori operator is taken from TRIQS, so both
# sides solve the same Hamiltonian.
#
# Self-tests, always run:
#   - phonon truncation: solved again with n_ph + n_ph_check levels, max differences reported;
#   - G_a(0) + G_a(beta) = -1 and chi_aa(0) = -G_a(beta) = <n_a>.
# With --V 0 and equal g for both orbitals, sum_a g_a n_a commutes with H, so the Lang-Firsov
# transform is exact and chi_ab(tau) must equal that of the fermions alone with the polaron
# shift -(sum_a g_a n_a)^2 / (2 omega_0^2); this is checked too.

import argparse
import os
import time
import numpy as np
from h5 import HDFArchive
from triqs.operators import n
import model as model_def

parser = argparse.ArgumentParser(description='ED reference: Kanamori impurity + bath sites + phonon.')
model_def.add_model_args(parser)
parser.add_argument('--n_ph', type=int, default=24, help='Phonon levels kept')
parser.add_argument('--n_ph_check', type=int, default=6, help='Also solve with n_ph + this many levels and report the difference')
parser.add_argument('--n_tau', type=int, default=401, help='Imaginary-time points on [0, beta]')
parser.add_argument('--n_iw', type=int, default=1025, help='Positive Matsubara frequencies for G(iw) and Sigma(iw)')
# No --target_n here: the mu bisection lives in calibrate_mu.py, which drives this script
# as a subprocess and reads the <n_a> line below. One implementation, not two.
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

n_so    = len(M.labels)  # impurity spin-orbitals: modes 0 .. n_so-1
n_modes = 2 * n_so       # bath site of spin-orbital a: mode n_so + a
spin_of_mode = [M.labels[j % n_so][0] for j in range(n_modes)]


def annihilators(n_modes):
    """Jordan-Wigner annihilation operators, mode 0 is the leftmost tensor factor."""
    lower, Z, I = np.array([[0., 1.], [0., 0.]]), np.diag([1., -1.]), np.eye(2)
    ops = []
    for j in range(n_modes):
        m = np.array([[1.]])
        for k in range(n_modes):
            m = np.kron(m, Z if k < j else (lower if k == j else I))
        ops.append(m)
    return ops


c     = annihilators(n_modes)
dim_f = 2**n_modes
occ   = np.array([[(s >> (n_modes - 1 - j)) & 1 for j in range(n_modes)] for s in range(dim_f)])
for j in range(n_modes):
    assert np.allclose(c[j].T @ c[j], np.diag(occ[:, j])), "Jordan-Wigner construction inconsistent with occupations"
mode_of = {label: a for a, label in enumerate(M.labels)}


def to_matrix(op):
    """Matrix of a TRIQS many-body operator on the impurity modes."""
    mat = np.zeros((dim_f, dim_f))
    for term, coeff in op:
        assert abs(np.imag(coeff)) < 1e-12
        m = np.eye(dim_f)
        for dagger, indices in term:
            cj = c[mode_of[tuple(indices)]]
            m = m @ (cj.T if dagger else cj)
        mat += np.real(coeff) * m
    return mat


H_f = to_matrix(M.h_int() - sum(M.mu[a] * n(*M.labels[a]) for a in range(n_so)))
for a in range(n_so):
    b = c[n_so + a]
    H_f += M.eps_bath * b.T @ b + M.V * (c[a].T @ b + b.T @ c[a])
F = occ[:, :n_so] @ M.g  # sum_a g_a n_a, diagonal in the occupation basis

# N_up and N_down, impurity and bath together, are conserved by every term: block-diagonalize
N_up = occ[:, [j for j in range(n_modes) if spin_of_mode[j] == 'up']].sum(axis=1)
N_dn = occ[:, [j for j in range(n_modes) if spin_of_mode[j] == 'down']].sum(axis=1)
block_keys = sorted(set(zip(N_up, N_dn)))
tau = np.linspace(0, M.beta, args.n_tau)
# Positive fermionic Matsubara frequencies; Sigma is reported on these.
w_n = (2 * np.arange(args.n_iw) + 1) * np.pi / M.beta
iw = 1j * w_n


def correlators(blocks):
    """chi_ab(tau), G_a(tau) and phonon diagnostics from the eigen-decomposition of every block."""
    E0 = min(E.min() for _, E, _, _ in blocks.values())
    Z  = sum(np.exp(-M.beta * (E - E0)).sum() for _, E, _, _ in blocks.values())
    chi, G = np.zeros((n_so, n_so, len(tau))), np.zeros((n_so, len(tau)))
    G_iw = np.zeros((n_so, len(iw)), dtype=complex)
    mean_phonons, top_level = 0.0, 0.0
    for key, (idx, E, U, n_ph) in blocks.items():
        weight_left  = np.exp(-np.outer(M.beta - tau, E - E0))  # e^{-(beta - tau) E_m}
        weight_right = np.exp(-np.outer(tau, E - E0))           # e^{-tau E_n}
        boltzmann    = np.exp(-M.beta * (E - E0))

        # n_a (x) 1_phonon in this block's eigenbasis; chi_ab = Tr[e^{-(beta-tau)H} n_a e^{-tau H} n_b] / Z
        dens = [U.T @ (np.repeat(occ[idx, a], n_ph)[:, None] * U) for a in range(n_so)]
        for a in range(n_so):
            for b in range(n_so):
                chi[a, b] += ((weight_left @ (dens[a] * dens[b])) * weight_right).sum(axis=1)

        # G_a(tau) = -sum_{m,n} e^{-(beta-tau) E_m} e^{-tau E_n} |<m|c_a|n>|^2 / Z, m one particle fewer
        for a in range(n_so):
            key_minus = (key[0] - 1, key[1]) if M.labels[a][0] == 'up' else (key[0], key[1] - 1)
            if key_minus not in blocks: continue
            idx_m, E_m, U_m, _ = blocks[key_minus]
            C = U_m.T @ np.kron(c[a][np.ix_(idx_m, idx)], np.eye(n_ph)) @ U
            G[a] -= ((np.exp(-np.outer(M.beta - tau, E_m - E0)) @ C**2) * weight_right).sum(axis=1)

            # Same Lehmann sum directly on the Matsubara axis:
            #   G_a(iw) = (1/Z) sum_{m,n} |<m|c_a|n>|^2 (e^{-beta E_m} + e^{-beta E_n})
            #                             / (iw + E_m - E_n)
            # Doing this instead of Fourier-transforming G(tau) avoids any discretisation
            # error, so the resulting Sigma is exact up to the phonon truncation -- which is
            # the whole point of having an ED reference to compare Sigma against.
            dE = E_m[:, None] - E[None, :]                       # E_m - E_n
            w8 = np.exp(-M.beta * (E_m - E0))[:, None] + np.exp(-M.beta * (E - E0))[None, :]
            G_iw[a] += ((C**2 * w8)[None, :, :] / (iw[:, None, None] + dE[None, :, :])).sum(axis=(1, 2))

        phonon_number = np.tile(np.arange(n_ph), len(idx))
        mean_phonons += boltzmann @ (phonon_number @ U**2)
        top_level    += boltzmann @ ((phonon_number == n_ph - 1) @ U**2)
    return chi / Z, G / Z, G_iw / Z, mean_phonons / Z, top_level / Z


def solve(n_ph):
    ph = np.arange(n_ph)
    d  = np.diag(np.sqrt(ph[1:]), 1)
    x  = (d + d.T) / np.sqrt(2 * M.omega_0)
    blocks = {}
    for key in block_keys:
        idx = np.where((N_up == key[0]) & (N_dn == key[1]))[0]
        H = (np.kron(H_f[np.ix_(idx, idx)], np.eye(n_ph)) + np.kron(np.eye(len(idx)), M.omega_0 * np.diag(ph))
             + np.kron(np.diag(F[idx]), x))
        E, U = np.linalg.eigh(H)
        blocks[key] = (idx, E, U, n_ph)
    return correlators(blocks)


start = time.time()
chi, G, G_iw, mean_phonons, top_level = solve(args.n_ph)
chi_check, G_check, _, _, _ = solve(args.n_ph + args.n_ph_check)
truncation_chi = np.abs(chi - chi_check).max()
truncation_G   = np.abs(G - G_check).max()
print(f"ED done in {time.time() - start:.1f} s, largest block {max(len(np.where((N_up == k[0]) & (N_dn == k[1]))[0]) for k in block_keys) * args.n_ph}")
print(f"Phonon truncation: <N_ph> = {mean_phonons:.3f}, weight in top level = {top_level:.2e}, "
      f"max|chi(n_ph) - chi(n_ph + {args.n_ph_check})| = {truncation_chi:.2e}, same for G: {truncation_G:.2e}")

occupations = -G[:, -1]
print("<n_a> =", np.round(occupations, 5), "(0.5 at half filling)")
print(f"max |G_a(0) + G_a(beta) + 1| = {np.abs(G[:, 0] + G[:, -1] + 1).max():.2e}, "
      f"max |chi_aa(0) - <n_a>| = {np.abs(np.diag(chi[:, :, 0]) - occupations).max():.2e}")

lang_firsov_check = None
if M.V == 0 and M.g_orb[0] == M.g_orb[1]:
    # Exact: the Lang-Firsov unitary commutes with every n_a, so chi is that of the shifted fermions
    E, U = np.linalg.eigh(H_f - np.diag(F**2) / (2 * M.omega_0**2))
    E = E - E.min()
    Z = np.exp(-M.beta * E).sum()
    dens = [U.T @ (occ[:, a][:, None] * U) for a in range(n_so)]
    wl, wr = np.exp(-np.outer(M.beta - tau, E)), np.exp(-np.outer(tau, E))
    chi_shifted = np.array([[((wl @ (dens[a] * dens[b])) * wr).sum(axis=1) / Z for b in range(n_so)] for a in range(n_so)])
    lang_firsov_check = np.abs(chi - chi_shifted).max()
    print(f"V = 0 Lang-Firsov check: max |chi_ED - chi_shifted_fermions| = {lang_firsov_check:.2e}")

# --------------------------------------------------------------------------- self-energy
#
# Sigma_a(iw) = G0_a(iw)^-1 - G_a(iw)^-1,   G0_a(iw)^-1 = iw + mu_a - Delta_a(iw)
#
# with Delta_a(iw) = V^2/(iw - eps_bath) -- the *same* hybridization the CTHYB side is
# handed, since the whole point of this model is that both sides solve one Hamiltonian.
# Everything here is diagonal in the spin-orbital index, so Sigma is a scalar per orbital.
#
# G_iw came from the Lehmann sum rather than from Fourier-transforming G(tau), so this
# Sigma carries no discretisation error: it is exact up to the phonon truncation, which is
# quoted above. That makes it a genuine reference curve for the QMC Sigma rather than
# another approximation to argue with.
delta_iw = M.V**2 / (iw - M.eps_bath)
Sigma_iw = np.zeros_like(G_iw)
for a in range(n_so):
    Sigma_iw[a] = (iw + M.mu[a] - delta_iw) - 1.0 / G_iw[a]

print(f"Sigma: Im<0 on the first 50 w_n for every orbital: "
      f"{bool(np.all(Sigma_iw[:, :50].imag < 0))}")
window = (w_n > 1.0) & (w_n < 8.0)
print("  Re Sigma(1<w<8) per orbital =", np.round(Sigma_iw[:, window].real.mean(axis=1), 4))

os.makedirs(args.out_dir, exist_ok=True)
filename = os.path.join(args.out_dir, f"ed_{M.tag()}_nph-{args.n_ph}.h5")
with HDFArchive(filename, 'w') as A:
    A['solver'] = 'ed'
    A['tau'] = tau
    A['chi'] = chi
    A['G'] = G
    A['w_n'] = w_n
    A['G_iw'] = G_iw
    A['Sigma'] = Sigma_iw
    A['labels'] = [f"{s},{o}" for s, o in M.labels]
    A['mu'] = M.mu
    A['density'] = occupations
    A['params'] = M.params()
    A['n_ph'] = args.n_ph
    A['truncation_chi'] = truncation_chi
    A['truncation_G'] = truncation_G
    A['ed_truncation'] = max(truncation_chi, truncation_G)
    A['mean_phonons'] = mean_phonons
    if lang_firsov_check is not None: A['lang_firsov_check'] = lang_firsov_check
print(f"Saved {filename}")
