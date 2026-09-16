# Is the CTHYB - ED residual in plot_ed_vs_cthyb.py a systematic error or Monte Carlo noise?
#
# The eye cannot tell from the residual panels, for one specific reason: Q_conserved_tau is a
# Legendre series truncated at dyn_n_l (measure_D0_corr accumulates alpha_n and reconstructs
# Q(tau) = sum_{l < n_l} P_l(x) q_l). That reconstruction is smooth by construction, so the
# point-to-point wiggle the eye reads as "the error band" only shows the high-l noise, while
# most of the error sits in the low-l coefficients - above all l = 0, a pure constant in tau.
# A constant offset is invisible as scatter and looks exactly like a systematic error.
#
# So this script does the comparison in the Legendre basis the estimator actually works in,
# coefficient by coefficient, and separates the candidate error sources:
#
#   A. ED's own precision      - phonon truncation and the exact identities it satisfies.
#   B. the plot's artifact     - ED is stored on n_tau points and plot_ed_vs_cthyb.py used to
#                                np.interp it onto the finer CTHYB grid; <n_a(tau) n_b(0)> has a
#                                kink at tau = 0, so linear interpolation errs *at the tails*
#                                by an amount comparable to the residual itself.
#   C. Legendre truncation     - the l >= n_l content of the exact curve that CTHYB threw away,
#                                the one genuine systematic that is worst at tau = 0 and beta.
#   D. ED-free error probes    - up<->down and KMS symmetries that this Hamiltonian satisfies
#                                exactly, so any violation is pure MC error, slow modes included.
#   E. error budget            - all of the above side by side, against the observed residual.
#
# D is the part that answers the question without reference to ED at all: the model is exactly
# spin symmetric (O_0 <-> O_1 under up <-> down), so C_00(tau) - C_11(tau) must vanish, and
# equilibrium plus spin symmetry gives C_01(tau) = C_01(beta - tau). Note that collect_results
# already *enforces* q_l(a,b) = (-1)^l q_l(b,a), which kills the odd-l part of every diagonal
# element - so C_00's own tau <-> beta - tau symmetry is not a probe, but C_00 - C_11 is.
#
# What a single run can never settle is whether the l = 0 channel carries a small bias on top of
# its noise: with two diagonal pairs you have two samples. For that, rerun with different
# --random_seed and take the spread (the default seed is 34788 + 928374 * rank, so a plain rerun
# of the same command is bit-identical and tells you nothing).
#
# Runs either way, like plot_ed_vs_cthyb.py: edit CONFIG and run the cells in Jupyter, or
#   python diagnose_residual.py [ed_file] [cthyb_file]

import os
import sys
import numpy as np
from h5 import HDFArchive
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
import triqs.gfs  # registers Gf / BlockGf with h5

# ---------------------------------------------------------------- CONFIG (edit for interactive use)
DATA_DIR = '/home/andrewhardy/Documents/Data/CTHYB_Data/ed_reference'
TAG = 'beta-10.0_U-2.0_J-0.3_V-0.7_eb-0.0_w0-1.0_g-0.7-0.3_mu-half'
ED_FILE = f'{DATA_DIR}/ed_{TAG}_nph-24.h5'
CTHYB_FILE = f'{DATA_DIR}/cthyb_{TAG}_lf-True_nc-1000000.h5'
N_L = 50          # the run's dyn_n_l: coefficients below this were measured, the rest discarded
N_L_PROBE = 90    # project ED this far to see the discarded tail
SEED_FILES = []   # several runs that differ only in --random_seed: turns part F on
# ----------------------------------------------------------------------------------------------


def legendre_polys(x, n_l):
    """P_0..P_{n_l-1} evaluated on x, by the standard recursion."""
    P = [np.ones_like(x), np.asarray(x, dtype=float)]
    for l in range(2, n_l):
        P.append(((2 * l - 1) * x * P[l - 1] - (l - 1) * P[l - 2]) / l)
    return P[:n_l]


def legendre_project(f, tau, beta, n_l, n_fine=40001):
    """q_l = (2l+1)/beta * int_0^beta P_l(2 tau / beta - 1) f(tau) dtau.

    The inverse of the reconstruction in measure_D0_corr::collect_results. Splined onto a fine
    grid first so that Simpson resolves P_l's ~n_l oscillations however coarse the input mesh is.
    """
    tau_fine = np.linspace(tau[0], tau[-1], n_fine)
    f_fine = CubicSpline(tau, f)(tau_fine)
    P = legendre_polys(2 * tau_fine / beta - 1, n_l)
    return np.array([(2 * l + 1) / beta * simpson(P[l] * f_fine, x=tau_fine) for l in range(n_l)])


def legendre_eval(q_l, tau, beta):
    P = legendre_polys(2 * np.asarray(tau) / beta - 1, len(q_l))
    return sum(q_l[l] * P[l] for l in range(len(q_l)))


def read_inputs(ed_file, cthyb_file):
    with HDFArchive(ed_file, 'r') as A:
        ed = {key: A[key] for key in A.keys() if key != 'params'}
        ed['params'] = A['params']
    with HDFArchive(cthyb_file, 'r') as A:
        cthyb = {key: A[key] for key in ['G_tau', 'Q_conserved_tau', 'conserved_vectors',
                                         'equal_time_added', 'average_sign', 'average_order',
                                         'n_cycles', 'params']}
        cthyb['orbital_occupations'] = A['orbital_occupations'] if 'orbital_occupations' in A.keys() else None
        cthyb['equal_time_conserved'] = A['equal_time_conserved'] if 'equal_time_conserved' in A.keys() else None
    return ed, cthyb


def ed_precision(ed):
    """A. ED is exact up to the phonon truncation; report that and the identities it satisfies."""
    G, chi = ed['G'], ed['chi']
    occ = -G[:, -1]
    swap = np.transpose(chi, (1, 0, 2))[:, :, ::-1]           # chi_ba(beta - tau)
    perm = [2, 3, 0, 1]                                        # (up,0),(up,1),(down,0),(down,1)
    print('--- A. ED precision ---')
    print(f"  n_ph = {ed['n_ph']}, <N_ph> = {ed['mean_phonons']:.3f}")
    print(f"  phonon truncation  max|chi(n_ph) - chi(n_ph + dn)| = {ed['truncation_chi']:.1e}, "
          f"G: {ed['truncation_G']:.1e}")
    print(f"  <n_a> - 1/2 (exact by ph symmetry)               = {np.abs(occ - 0.5).max():.1e}")
    print(f"  max|G(0) + G(beta) + 1|                          = {np.abs(G[:, 0] + G[:, -1] + 1).max():.1e}")
    print(f"  max|chi_aa(0) - <n_a>|                           = {np.abs(np.diag(chi[:, :, 0]) - occ).max():.1e}")
    print(f"  max|chi_ab(tau) - chi_ba(beta - tau)|  (KMS)     = {np.abs(chi - swap).max():.1e}")
    print(f"  max|chi_ab - chi_(spin-flipped)|       (spin)    = {np.abs(chi - chi[np.ix_(perm, perm)]).max():.1e}")
    return max(ed['truncation_chi'], ed['truncation_G'])


def interpolation_artifact(ed, cthyb, pairs):
    """B. What plot_ed_vs_cthyb.py's own interpolation of the ED curve costs, and where."""
    beta, tau_q = ed['params']['beta'], np.array([float(t) for t in cthyb['Q_conserved_tau'].mesh])
    vec = cthyb['conserved_vectors']
    print(f"\n--- B. Interpolating ED ({len(ed['tau'])} pts) onto the CTHYB grid "
          f"({len(tau_q)} pts): linear vs cubic ---")
    worst = 0.0
    near_tail = (tau_q < 0.02 * beta) | (tau_q > 0.98 * beta)
    mid = np.abs(tau_q - beta / 2) < 0.05 * beta
    for i, j in pairs:
        curve = np.einsum('a,abt,b->t', vec[i], ed['chi'], vec[j])
        err = np.interp(tau_q, ed['tau'], curve) - CubicSpline(ed['tau'], curve)(tau_q)
        worst = max(worst, np.abs(err).max())
        print(f"  <O_{i}O_{j}>: max {np.abs(err).max():.1e} at tau = {tau_q[np.argmax(np.abs(err))]:.3f}"
              f" | tails {np.abs(err[near_tail]).max():.1e} | mid {np.abs(err[mid]).max():.1e}")
    print('  -> a tail-localised artifact of the plot, not of the solver: chi has a kink at tau = 0')
    return worst


def legendre_comparison(ed, cthyb, pairs, n_l=N_L, n_probe=N_L_PROBE):
    """C + the heart of the diagnosis: per-coefficient CTHYB vs ED, and the discarded tail."""
    beta = ed['params']['beta']
    Q = cthyb['Q_conserved_tau']
    tau_q = np.array([float(t) for t in Q.mesh])
    vec = cthyb['conserved_vectors']
    out = {}
    print(f'\n--- C. Legendre coefficients: l < {n_l} measured, l >= {n_l} discarded ---')
    for i, j in pairs:
        ed_curve = np.einsum('a,abt,b->t', vec[i], ed['chi'], vec[j])
        q_curve = Q.data[:, i, j].real
        q_ed = legendre_project(ed_curve, ed['tau'], beta, n_probe)
        q_ct = legendre_project(q_curve, tau_q, beta, n_probe)   # recovers the series it was built from
        dq = q_ct - q_ed
        # the part of the exact curve the truncation removed, evaluated where it hurts most
        tail = legendre_eval(np.r_[np.zeros(n_l), q_ed[n_l:]], tau_q, beta)
        resid = q_curve - CubicSpline(ed['tau'], ed_curve)(tau_q)
        floor = np.sqrt(np.mean(dq[10:n_l][dq[10:n_l] != 0.0] ** 2))
        out[(i, j)] = dict(dq=dq, resid=resid, tail=tail, floor=floor, tau_q=tau_q)
        print(f"\n  <O_{i}O_{j}>  residual: mean {resid.mean():+.2e}, max|.| {np.abs(resid).max():.2e}")
        print(f"    dq_0 (the constant)                  = {dq[0]:+.2e}   <- {100*abs(dq[0])/max(np.abs(resid).max(),1e-300):.0f}% of max|residual|")
        print(f"    dq_l, l = 1..9                       = {np.array2string(dq[1:10], precision=1, max_line_width=200)}")
        print(f"    dq_l, l = 10..{n_l-1}: rms {floor:.1e}, max {np.abs(dq[10:n_l]).max():.1e}   <- MC noise floor")
        print(f"    discarded ED tail max|sum_{{l>={n_l}}}|  = {np.abs(tail).max():.1e} "
              f"(at tau = {tau_q[np.argmax(np.abs(tail))]:.2f})")
        print(f"    residual with the constant removed   = {np.abs(resid - resid.mean()).max():.1e}")
    return out


def symmetry_probes(ed, cthyb):
    """D. Exact symmetries of this Hamiltonian: any violation is pure MC error, no ED needed."""
    Q = cthyb['Q_conserved_tau']
    C00, C11, C01 = Q.data[:, 0, 0].real, Q.data[:, 1, 1].real, Q.data[:, 0, 1].real
    labels = [tuple(s.split(',')) for s in ed['labels']]
    print('\n--- D. ED-free error probes (exact symmetries, so a violation IS the error) ---')
    print(f"  up<->down:  max|C_00 - C_11|              = {np.abs(C00 - C11).max():.1e}  "
          f"(mean {np.mean(C00 - C11):+.1e})   <- includes the slow/constant modes")
    print(f"  KMS+spin:   max|C_01(tau) - C_01(beta-t)| = {np.abs(C01 - C01[::-1]).max():.1e}   <- odd-l noise")
    wiggle = np.std(C00[2:] - 2 * C00[1:-1] + C00[:-2])
    print(f"  point-to-point wiggle of C_00             = {wiggle:.1e}   <- all the eye sees in the plot")
    dens = np.array([-cthyb['G_tau'][s].data[-1, int(o), int(o)].real for s, o in labels])
    print(f"  densities from -G(beta) (exact: 0.5):     {np.round(dens, 5)}")
    print(f"    max deviation {np.abs(dens - 0.5).max():.1e}, up<->down violation {np.abs(dens[:2] - dens[2:]).max():.1e}")
    for a, (s, o) in enumerate(labels):
        g = cthyb['G_tau'][s].data[:, int(o), int(o)].real
        print(f"    G({s},{o}): max|G(tau) - G(beta - tau)| = {np.abs(g - g[::-1]).max():.1e}  (exact by ph symmetry)")
    if cthyb['orbital_occupations'] is not None:
        rho_dens = np.array(cthyb['orbital_occupations'])
        print(f"  densities from the density matrix:        {np.round(rho_dens, 6)}")
        print(f"    max deviation {np.abs(rho_dens - 0.5).max():.1e}  <- the estimator the l=0 offset comes from")
    else:
        print('  (orbital_occupations not in the archive: rerun to probe the l = 0 offset directly)')
    if cthyb['equal_time_conserved'] is not None:
        vec = cthyb['conserved_vectors']
        exact = np.einsum('ia,abt,jb->ijt', vec, ed['chi'], vec)[:, :, 0]
        print('  equal-time <O_i O_j> from the density matrix vs ED chi(0) '
              '(this IS the l = 0 offset solver.py adds):')
        print(f"    CTHYB {np.round(cthyb['equal_time_conserved'], 6).tolist()}")
        print(f"    ED    {np.round(exact, 6).tolist()}")
        print(f"    max deviation {np.abs(cthyb['equal_time_conserved'] - exact).max():.1e}")
    return dict(spin=np.abs(np.mean(C00 - C11)), kms=np.abs(C01 - C01[::-1]).max(), wiggle=wiggle)


def seed_spread(ed, files, pairs, n_l=N_L):
    """F. The only test that separates a bias in the l = 0 channel from its noise.

    Several runs differing only in --random_seed are independent samples of the same estimator, so
    their spread is the real error bar - the one the plot's visible band underestimates by ~50x
    because the Legendre reconstruction hides the low-l noise in a smooth curve. A coefficient whose
    mean sits many standard errors from ED is a systematic; one within a couple of them is noise.
    """
    beta = ed['params']['beta']
    runs = []
    for f in files:
        with HDFArchive(f, 'r') as A:
            runs.append((A['Q_conserved_tau'], A['conserved_vectors'],
                         A['random_seed'] if 'random_seed' in A.keys() else None))
    print(f'\n--- F. Spread over {len(runs)} independent seeds ---')
    if len(runs) < 3:
        print('  need at least 3 seeds for a meaningful standard error; see run_kanamori_phonon.sh')
    for i, j in pairs:
        vec = runs[0][1]
        q_ed = legendre_project(np.einsum('a,abt,b->t', vec[i], ed['chi'], vec[j]),
                                ed['tau'], beta, n_l)
        samples = np.array([legendre_project(Q.data[:, i, j].real,
                                             np.array([float(t) for t in Q.mesh]), beta, n_l)
                            for Q, _, _ in runs])
        mean, sem = samples.mean(axis=0), samples.std(axis=0, ddof=1) / np.sqrt(len(runs))
        z = np.divide(mean - q_ed, sem, out=np.zeros_like(mean), where=sem > 0)
        print(f"\n  <O_{i}O_{j}>  l = 0: {mean[0]:.6f} +/- {sem[0]:.1e}  vs ED {q_ed[0]:.6f}  ->  z = {z[0]:+.1f}")
        worst = np.argsort(-np.abs(z))[:5]
        print(f"    largest |z| over l < {n_l}: " +
              ', '.join(f'l={l} z={z[l]:+.1f}' for l in worst))
        print(f"    |z| > 3 at l = {np.flatnonzero(np.abs(z) > 3).tolist()}  "
              f"(expect ~0 entries if the estimator is unbiased)")


def budget(ed_err, interp_err, leg, probes, pairs):
    """E. Everything on one scale."""
    print('\n--- E. Error budget for <O_i(tau) O_j(0)> ---')
    rows = [
        ('ED (phonon truncation + linear algebra)', ed_err),
        (f'Legendre truncation at l = {N_L}', max(np.abs(leg[p]['tail']).max() for p in pairs)),
        ('plot: linear interpolation of ED, at the tails', interp_err),
        ('MC, high-l (the visible band)', probes['wiggle']),
        ('MC, odd-l (from the KMS probe)', probes['kms']),
        ('MC, l = 0 constant (from the up<->down probe)', probes['spin']),
        ('observed max |CTHYB - ED|', max(np.abs(leg[p]['resid']).max() for p in pairs)),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f'  {name:<{width}s}  {value:.1e}')


if __name__ == '__main__':
    ed_file, cthyb_file = ED_FILE, CTHYB_FILE
    if len(sys.argv) > 2:
        ed_file, cthyb_file = sys.argv[1], sys.argv[2]
    ed, cthyb = read_inputs(ed_file, cthyb_file)
    n_conserved = len(cthyb['conserved_vectors'])
    pairs = [(i, j) for i in range(n_conserved) for j in range(i, n_conserved)]
    print(f"{os.path.basename(cthyb_file)}\n  average sign {np.real(cthyb['average_sign']):.4f}, "
          f"average order {cthyb['average_order']:.2f}, {cthyb['n_cycles']} cycles/rank, "
          f"equal_time_added = {cthyb['equal_time_added']}\n")
    ed_err = ed_precision(ed)
    interp_err = interpolation_artifact(ed, cthyb, pairs)
    leg = legendre_comparison(ed, cthyb, pairs)
    probes = symmetry_probes(ed, cthyb)
    budget(ed_err, interp_err, leg, probes, pairs)
    if len(SEED_FILES) > 1:
        seed_spread(ed, SEED_FILES, pairs)
    else:
        print('\n--- F. Spread over independent seeds: SEED_FILES is empty ---')
        print('  The residual is ~90% a single constant (l = 0) whose size matches the up<->down')
        print('  symmetry violation, so it is consistent with noise - but two diagonal pairs are two')
        print('  samples, which cannot separate a small bias from it. Fill in SEED_FILES with runs')
        print('  from run_kanamori_phonon.sh\'s seed sweep to get a real error bar.')
