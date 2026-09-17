# Analysis for the two-patch DCA / real-space S.S benchmark. Runs top to bottom in a
# notebook (paste it into a cell, or `python plot_dca_spin_spin.py`); edit the two paths
# just below. Figures are written next to the LF_TRUE file and also shown.
#
# What it produces:
#   1. G_tau for each patch, lang_firsov True vs False. These are the same physics by two
#      different routes -- the analytic projector split plus stochastic residual, versus
#      everything stochastic -- so the difference is the headline internal check.
#   2. <S_tot^z(tau) S_tot^z(0)> from the O_tau estimator, both routes.
#   3. Site-resolved <S_i(tau).S_j(0)>, reconstructed from the coupling-derivative
#      estimator. O_tau cannot measure these (the site-basis spin operators do not commute
#      with h_loc), but the per-vertex correlators can be contracted back into them:
#
#          <S_i(tau).S_j(0)> = sum_{mn} a_m b_n <M_m(tau) N_n(0)>
#
#      where a_m, b_n are exactly the expansion coefficients dca_model used to register
#      the vertices. Needs the lang_firsov=False run: only stochastic vertices are
#      measured, so anything routed analytically is missing from the sum (the script says
#      so rather than silently under-reporting).
#   4. Expansion order histograms, including the dynamical-vertex order.

import os
import numpy as np
import matplotlib.pyplot as plt
# These imports look unused, but they register the Gf / BlockGf / Histogram readers
# with the h5 format registry. Without them HDFArchive hands back raw
# HDFArchiveGroups and every .mesh / .data access below fails.
from triqs.gfs import Gf, BlockGf, MeshImTime
from triqs.stat.histograms import Histogram
from h5 import HDFArchive

# ---------------------------------------------------------------------------- paths ----
DATA = '/home/andrewhardy/Documents/Data/CTHYB_Data/dca_spin_spin'
LF_TRUE = os.path.join(DATA, 'cthyb_dca_beta-10.0_t-0.25_tp-0.0_U-2.0_Jintra-0.0_Jinter-0.5'
                             '_w0-1.0_mu-1.0_bath-dca_rot-site_lf-True_nc-1000000.h5')
LF_FALSE = LF_TRUE.replace('lf-True', 'lf-False')
# ---------------------------------------------------------------------------------------

INK = '#1a1a1a'
COLORS = ['#3b6ea5', '#c1666b', '#5b8c5a', '#b58b4c']


def load(path):
    if not os.path.exists(path):
        print(f"missing: {path}")
        return None
    with HDFArchive(path, 'r') as A:
        keys = list(A.keys())
        d = dict(params=A['params'], G_tau=A['G_tau'], average_sign=A['average_sign'],
                 average_order=A['average_order'], lang_firsov=A['lang_firsov'],
                 n_cycles=A['n_cycles'], fillings=A['fillings'], total_filling=A['total_filling'],
                 mu_orbital=A['mu_orbital'], conserved=A['conserved_operators'],
                 O_tau=A['O_tau'] if 'O_tau' in keys else None,
                 Q_conserved_tau=A['Q_conserved_tau'] if 'Q_conserved_tau' in keys else None,
                 perturbation_order=A['perturbation_order'] if 'perturbation_order' in keys else None,
                 perturbation_order_dyn=A['perturbation_order_dyn'] if 'perturbation_order_dyn' in keys else None,
                 vertex_labels=[list(x) for x in A['vertex_labels']] if 'vertex_labels' in keys else None,
                 vertex_corr=A['vertex_corr'] if 'vertex_corr' in keys else None,
                 vertex_tau=A['vertex_tau'] if 'vertex_tau' in keys else None)
    return d


runs = {'lf=True': load(LF_TRUE), 'lf=False': load(LF_FALSE)}
runs = {k: v for k, v in runs.items() if v is not None}
if not runs:
    raise SystemExit('No input files found -- edit DATA / LF_TRUE at the top of this script.')

print('=' * 78)
for name, r in runs.items():
    p = r['params']
    print(f"{name:9s}  sign {np.real(r['average_sign']):.4f}  order {np.real(r['average_order']):.2f}  "
          f"<N> {r['total_filling']:.4f}  cycles {r['n_cycles']}")
    print(f"           U {p['U']}, J_intra {p['J_intra']}, J_inter {p['J_inter']}, beta {p['beta']}, "
          f"eps_patch {np.round(p['eps_patch'], 4)}, mu {np.round(r['mu_orbital'], 4)}")
    print(f"           conserved density combinations: {r['conserved'] if len(r['conserved']) else 'none'}")
print('=' * 78)


# ---------------------------------------------------------------------------------------
# 1 + 2: G_tau and the total-spin correlator, analytic-split vs fully stochastic
# ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

for k, (name, r) in enumerate(runs.items()):
    style = dict(linewidth=2, alpha=0.9) if k == 0 else dict(linewidth=1.2, linestyle=(0, (4, 2)))
    for K, patch in enumerate(['even', 'odd']):
        g = r['G_tau']['up']
        tau = np.array([float(t) for t in g.mesh.values()])
        ax[0, 0].plot(tau, g.data[:, K, K].real, color=COLORS[K], label=f'{patch}, {name}', **style)
    if r['O_tau'] is not None:
        O = r['O_tau']
        tau_O = np.array([float(t) for t in O.mesh.values()])
        ax[0, 1].plot(tau_O, O.data.real.squeeze(), color=COLORS[2 + k],
                      label=f'{name}', **style)

ax[0, 0].set_xlabel(r'$\tau$'); ax[0, 0].set_ylabel(r'$G_K(\tau)$')
ax[0, 0].set_title('Patch Green functions', fontsize=10)
ax[0, 1].set_xlabel(r'$\tau$'); ax[0, 1].set_ylabel(r'$\langle S^z_{\rm tot}(\tau) S^z_{\rm tot}(0)\rangle$')
ax[0, 1].set_title(r'Total-spin correlator ($O_\tau$ estimator)', fontsize=10)

# The internal cross-check: the two routes must agree within Monte Carlo error
if len(runs) == 2:
    a, b = runs['lf=True'], runs['lf=False']
    dG = max(np.abs(a['G_tau']['up'].data[:, K, K].real - b['G_tau']['up'].data[:, K, K].real).max()
             for K in range(2))
    print(f"\nCROSS-CHECK  max |G_tau(lf=True) - G_tau(lf=False)| = {dG:.5f}")
    if a['O_tau'] is not None and b['O_tau'] is not None:
        dO = np.abs(a['O_tau'].data.real - b['O_tau'].data.real).max()
        print(f"CROSS-CHECK  max |O_tau(lf=True)  - O_tau(lf=False)|  = {dO:.5f}")
    print("             These two routes share no code path for the analytic part, so\n"
          "             agreement here is a real test of the projector split.")
    dmu = np.abs(np.asarray(a['mu_orbital']) - np.asarray(b['mu_orbital'])).max()
    if dmu > 1e-10:
        print(f"WARNING      the two runs used DIFFERENT mu (max difference {dmu:.2e}); they are not the\n"
              f"             same Hamiltonian, so the comparison above is not a valid cross-check.")

# ---------------------------------------------------------------------------------------
# 3: site-resolved <S_i(tau) . S_j(0)> from the coupling-derivative estimator
# ---------------------------------------------------------------------------------------
def site_correlators(run):
    """Contract the measured per-vertex correlators into <S_i(tau).S_j(0)>.

    Rebuilds the expansion coefficients from dca_model with the *same* parameters the run
    used, so the contraction cannot drift out of sync with what was registered.
    """
    import argparse
    import dca_model as model_def
    from dca_model import expand_retarded_product, key_to_string, N_PATCH

    p = run['params']
    ns = argparse.Namespace(beta=p['beta'], t=p['t'], tp=p['tp'], U=p['U'], J_intra=p['J_intra'],
                            J_inter=p['J_inter'], omega_0=p['omega_0'], mu=p['mu'], bath=p['bath'],
                            V=p['V'], eps_bath=p['eps_bath'], rotation=p['rotation'],
                            n_k=1000, n_bins=50)
    M = model_def.Model(ns)

    measured = {(l1, l2): run['vertex_corr'][t]
                for t, (l1, l2) in enumerate(run['vertex_labels'])}
    # Masked points are stored as exactly 0 by the estimator; treat them as missing.
    out, warnings = {}, []
    for i in range(N_PATCH):
        for j in range(N_PATCH):
            coeffs = {}
            expand_retarded_product(M.Sz_site[i], M.Sz_site[j], 1.0, coeffs)
            expand_retarded_product(M.Sp_site[i], M.Sm_site[j], 0.5, coeffs)
            expand_retarded_product(M.Sm_site[i], M.Sp_site[j], 0.5, coeffs)
            total, missing = None, 0.0
            for (k1, k2), a in coeffs.items():
                if abs(a) < 1e-12:
                    continue
                label = (key_to_string(k1), key_to_string(k2))
                if label not in measured:
                    missing += abs(a)
                    continue
                contribution = a * measured[label]
                total = contribution if total is None else total + contribution
            if total is not None:
                out[(i, j)] = total
            if missing > 1e-9:
                warnings.append(f"S_{i+1}.S_{j+1}: |coefficient| {missing:.3f} not measured "
                                f"(routed analytically, or estimator-masked)")
    return out, warnings


chi_site, chi_warnings = {}, []
source = runs.get('lf=False') or next(iter(runs.values()))
if source['vertex_corr'] is not None:
    chi_site, chi_warnings = site_correlators(source)
    tau_v = source['vertex_tau']
    for k, ((i, j), curve) in enumerate(sorted(chi_site.items())):
        if i > j:
            continue  # symmetric
        ax[1, 0].plot(tau_v, curve, color=COLORS[k % len(COLORS)], linewidth=1.6,
                      label=rf'$\langle S_{i+1}\!\cdot\!S_{j+1}\rangle$')
    print(f"\nSite-resolved correlators reconstructed from {len(source['vertex_labels'])} "
          f"measured vertices ({'lf=False' if source is runs.get('lf=False') else 'lf=True'} run)")
    for w in chi_warnings:
        print('  WARNING ' + w)
    if not chi_warnings:
        print('  every contributing vertex was measured -- the reconstruction is complete')
else:
    ax[1, 0].text(0.5, 0.5, 'no vertex correlators in file\n(needs measure_D0_corr)',
                  ha='center', va='center', transform=ax[1, 0].transAxes, color=INK)

ax[1, 0].set_xlabel(r'$\tau$'); ax[1, 0].set_ylabel(r'$\langle S_i(\tau)\cdot S_j(0)\rangle$')
ax[1, 0].set_title('Site-resolved spin correlators (coupling-derivative estimator)', fontsize=10)

# ---------------------------------------------------------------------------------------
# 4: expansion orders
# ---------------------------------------------------------------------------------------
for k, (name, r) in enumerate(runs.items()):
    for key, ls in (('perturbation_order', '-'), ('perturbation_order_dyn', '--')):
        h = r[key]
        if h is None:
            continue
        # perturbation_order is stored per hybridization block (a dict of histograms);
        # perturbation_order_dyn is a single histogram. Sum the blocks so each curve is
        # one distribution.
        hists = list(h.values()) if isinstance(h, dict) else [h]
        data = sum(np.asarray(x.data, dtype=float) for x in hists)
        norm = data.sum()
        if norm > 0:
            ax[1, 1].plot(np.arange(len(data)), data / norm, color=COLORS[k], linestyle=ls,
                          linewidth=1.4, label=f"{name} {'hyb' if 'dyn' not in key else 'dyn'}")
ax[1, 1].set_xlabel('expansion order'); ax[1, 1].set_ylabel('probability')
ax[1, 1].set_title('Expansion orders (solid: hybridization, dashed: dynamical)', fontsize=10)
ax[1, 1].set_yscale('log')
ax[1, 1].set_xlim(0,50)
for a_ in ax.flat:
    a_.legend(fontsize=7, frameon=False)
    a_.grid(alpha=0.25, linewidth=0.5)

out_png = os.path.splitext(LF_TRUE)[0] + '_dca_summary.png'
try:
    fig.savefig(out_png, dpi=150)
    print(f"\nFigure written to {out_png}")
except OSError as e:
    print(f"\nCould not write the figure ({e}); showing it instead")
plt.show()
