# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# A retarded density-density coupling that is not symmetric about beta/2, D(tau) = -g^2 exp(-omega_0 tau) on
# (n_up, n_dn), through Lang-Firsov. (A real coupling between hermitian densities is always symmetric,
# D_ab(tau) = D_ba(tau) = D_ab(beta - tau), so this is a check of the bookkeeping, not a physical model.)
#
# Each vertex is 1/2 int int D(tau - tau') n_up(tau) n_dn(tau'), so a kink of up at tau and a kink of dn at tau'
# interact through [K_up,dn(t) + K_dn,up(beta - t)] / 2, t = tau - tau', whose odd Legendre part does not vanish here,
# and the equal-time term of h_loc is (1/2) int D, not -K'(0). The kernel used to be (K_ab + K_ba) / 2 folded at
# beta/2 and the shift -K'(0), both right only for symmetric couplings.
#
#   1. One way, D on (up, dn), is the same action as D/2 on (up, dn) plus D(beta - tau)/2 on (dn, up): the same kernel
#      table and the same shift, so with the same seed the same Markov chain, to round-off.
#   2. Both ways, Lang-Firsov must agree with the stochastic expansion (lang_firsov=False), which samples D itself and
#      builds no kernel: G_l within Z_MAX standard errors, the error of each run being the difference of two halves
#      with independent seeds, pooled over the lowest L_MAX coefficients. The same file under mpirun on many ranks
#      (each rank does the full cycles) is the high-statistics check.

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Gf, inverse, iOmega_n, Fourier, MeshImFreq
from triqs.operators import n
from triqs_cthyb import Solver

beta, U, mu, V, eps = 5.0, 2.0, 1.0, 1.0, 0.5
omega_0, g = 1.0, 0.8
n_iw, n_tau = 200, 1001
L_MAX = 8      # G_l compared for l < L_MAX
Z_MAX = 5.0    # agreement within Z_MAX standard errors

tau = np.linspace(0.0, beta, n_tau)
D = -g**2 * np.exp(-omega_0 * tau)
D_reflected = D[::-1]  # D(beta - tau)

one_way = [(D, n('up', 0), n('dn', 0))]
both_ways = [(0.5 * D, n('up', 0), n('dn', 0)), (0.5 * D_reflected, n('dn', 0), n('up', 0))]


def solve(registrations, n_cycles, seed, lang_firsov=True):
    S = Solver(beta=beta, gf_struct=[('up', 1), ('dn', 1)], n_iw=n_iw, n_tau=n_tau, n_tau_bosonic=n_tau,
               delta_interface=True)
    delta = Gf(mesh=MeshImFreq(beta, 'Fermion', n_iw), target_shape=[1, 1])
    delta << V**2 * inverse(iOmega_n - eps)
    for _, d in S.Delta_tau:
        d << Fourier(delta)
    for coupling, op1, op2 in registrations:
        S.add_dyn_int(coupling, op1, op2)
    S.solve(h_int=U * n('up', 0) * n('dn', 0), h_loc0=-mu * (n('up', 0) + n('dn', 0)),
            n_cycles=n_cycles, n_warmup_cycles=max(500, n_cycles // 20), length_cycle=50,
            random_seed=seed + 928374 * mpi.rank, lang_firsov=lang_firsov,
            measure_density_matrix=True, use_norm_as_weight=True, measure_G_tau=True, measure_G_l=True, verbosity=0)
    return S


# 1. One way and both ways, same seed
S_one, S_both = (solve(r, n_cycles=3000, seed=8471) for r in (one_way, both_ways))
for (bl, g1), (_, g2) in zip(S_one.G_tau, S_both.G_tau):
    np.testing.assert_allclose(g1.data, g2.data, rtol=0, atol=1e-10, err_msg=f"G_tau[{bl}]")
for rho1, rho2 in zip(S_one.density_matrix, S_both.density_matrix):
    np.testing.assert_allclose(rho1, rho2, rtol=0, atol=1e-10)

# 2. Lang-Firsov against the stochastic expansion, two independent halves each
runs = {name: [solve(both_ways, n_cycles=50000, seed=1001 + 2 * k + h, lang_firsov=(name == "Lang-Firsov")) for h in range(2)]
        for k, name in enumerate(("Lang-Firsov", "stochastic"))}


def mean_and_error(halves):
    """G_l for l < L_MAX of both blocks, (2, L_MAX), the mean of the two halves and its standard error"""
    a, b = (np.array([S.G_l[bl].data[:L_MAX, 0, 0].real for bl in ('up', 'dn')]) for S in halves)
    return 0.5 * (a + b), np.sqrt(np.mean((0.5 * (a - b))**2))


if mpi.is_master_node():
    for name, halves in runs.items():
        for S in halves:
            assert S.average_sign > 0.0, "%s: non-positive average sign %s" % (name, S.average_sign)
    g_lf, error_lf = mean_and_error(runs["Lang-Firsov"])
    g_st, error_st = mean_and_error(runs["stochastic"])
    difference = np.abs(g_lf - g_st)
    error = np.hypot(error_lf, error_st)
    bl, l = np.unravel_index(difference.argmax(), difference.shape)
    mpi.report("Lang-Firsov - stochastic: largest |dG_l| %.5f (%s, l = %d), standard error %.5f"
               % (difference.max(), ('up', 'dn')[bl], l, error))
    assert difference.max() < Z_MAX * error, \
        "Lang-Firsov and stochastic G_l differ by %.5f at l = %d, more than %.1f standard errors (%.5f)" % (difference.max(), l, Z_MAX, error)
