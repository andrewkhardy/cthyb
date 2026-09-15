# Compare a CTHYB run (cthyb_kanamori_phonon.py) against the ED reference (ed_kanamori_phonon.py)
# for the same model:
#   left:  G_a(tau) for every spin-orbital, and CTHYB - ED;
#   right: <O_i(tau) O_j(0)> for the conserved density combinations CTHYB measured
#          (Q_conserved_tau), against the same contraction of the ED chi_ab, and CTHYB - ED.
# If the CTHYB run had no density matrix, its Q_conserved_tau lacks the equal-time constant
# <O_i O_j>; both curves are then shown relative to their value at beta/2.
#
#   python plot_ed_vs_cthyb.py data/ed_<tag>_nph-24.h5 data/cthyb_<tag>_lf-True_nc-1000000.h5

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from h5 import HDFArchive
import triqs.gfs  # registers Gf / BlockGf with h5

parser = argparse.ArgumentParser(description='Plot CTHYB against the ED reference.')
parser.add_argument('ed_file')
parser.add_argument('cthyb_file')
parser.add_argument('--out', default=None, help='Output image (default: next to cthyb_file, .png)')
parser.add_argument('--g_bins', type=int, default=100,
                    help='CTHYB G_tau is a per-bin histogram: compare averages over this many tau bins')
args = parser.parse_args()

# Categorical slots 1-4, fixed order; series are also told apart by legend, and source by line style
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'

with HDFArchive(args.ed_file, 'r') as A:
    ed = {key: A[key] for key in ['tau', 'chi', 'G', 'labels', 'params']}
with HDFArchive(args.cthyb_file, 'r') as A:
    G_tau, Q_conserved_tau = A['G_tau'], A['Q_conserved_tau']
    conserved_vectors, equal_time_added = A['conserved_vectors'], A['equal_time_added']
    lang_firsov, average_sign, n_cycles = A['lang_firsov'], A['average_sign'], A['n_cycles']
    cthyb_params = A['params']
    keys = list(A.keys())
    dyn_pairs = A['dyn_vertex_pairs'] if 'dyn_vertex_pairs' in keys else None
    dyn_corr = A['dyn_vertex_corr'] if 'dyn_vertex_corr' in keys else None
    dyn_tau = A['dyn_vertex_tau'] if 'dyn_vertex_tau' in keys else None

for key, value in ed['params'].items():
    if not np.isclose(value, cthyb_params[key]):
        raise ValueError(f"ED and CTHYB files are for different models: {key} = {value} vs {cthyb_params[key]}")
beta = ed['params']['beta']
labels = [tuple(label.split(',')) for label in ed['labels']]


def tau_of(gf):
    return np.linspace(0, beta, len(gf.mesh))


def bin_average(values, n_bins):
    return np.array([chunk.mean() for chunk in np.array_split(values, n_bins)])


def combination_name(v):
    terms = [(f"{c:g}" if abs(c - 1) > 1e-12 else "") + f"n({s},{o})" for c, (s, o) in zip(v, labels) if abs(c) > 1e-12]
    return " + ".join(terms)


def style_axis(ax, ylabel):
    ax.set_xlabel(r'$\tau$', color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlim(0, beta)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)


fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]), constrained_layout=True)
summary = []

# Left: G_a(tau)
for a, (s, o) in enumerate(labels):
    tau_c = tau_of(G_tau[s])
    # Same bins for both: CTHYB's histogram, and the (smooth) ED curve interpolated onto its mesh
    tau_b = bin_average(tau_c, args.g_bins)
    g_cthyb = bin_average(G_tau[s].data[:, int(o), int(o)].real, args.g_bins)
    g_ed = bin_average(np.interp(tau_c, ed['tau'], ed['G'][a]), args.g_bins)
    color = SERIES_COLORS[a]
    axes[0, 0].plot(ed['tau'], ed['G'][a], color=color, linewidth=2, label=f'ED  n({s},{o})')
    axes[0, 0].plot(tau_b, g_cthyb, color=color, linewidth=1, linestyle=(0, (4, 2)), label=f'CTHYB  n({s},{o})')
    residual = g_cthyb - g_ed
    axes[1, 0].plot(tau_b, residual, color=color, linewidth=1)
    summary.append(f"G({s},{o}): max |CTHYB - ED| over {args.g_bins} bins = {np.abs(residual).max():.4f}")
style_axis(axes[0, 0], r'$G_a(\tau)$')
style_axis(axes[1, 0], 'CTHYB $-$ ED')
axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
axes[0, 0].set_title(f'Green function (CTHYB averaged over {args.g_bins} bins)', color=INK, loc='left')

# Right: <O_i(tau) O_j(0)>
relative = not equal_time_added
tau_q = tau_of(Q_conserved_tau)
mid_ed, mid_q = np.argmin(np.abs(ed['tau'] - beta / 2)), np.argmin(np.abs(tau_q - beta / 2))
pairs = [(i, j) for i in range(len(conserved_vectors)) for j in range(i, len(conserved_vectors))]
for k, (i, j) in enumerate(pairs):
    ed_curve = np.einsum('a,abt,b->t', conserved_vectors[i], ed['chi'], conserved_vectors[j])
    q_curve = Q_conserved_tau.data[:, i, j].real
    if relative:
        ed_curve, q_curve = ed_curve - ed_curve[mid_ed], q_curve - q_curve[mid_q]
    color = SERIES_COLORS[k % len(SERIES_COLORS)]
    axes[0, 1].plot(ed['tau'], ed_curve, color=color, linewidth=2, label=fr'ED  $\langle O_{i} O_{j}\rangle$')
    axes[0, 1].plot(tau_q, q_curve, color=color, linewidth=1, linestyle=(0, (4, 2)), label=fr'CTHYB  $\langle O_{i} O_{j}\rangle$')
    residual = q_curve - np.interp(tau_q, ed['tau'], ed_curve)
    axes[1, 1].plot(tau_q, residual, color=color, linewidth=1)
    summary.append(f"<O_{i} O_{j}>: max |CTHYB - ED| = {np.abs(residual).max():.4f}")
ylabel = r'$\langle O_i(\tau) O_j(0)\rangle$' + (r' $-$ value at $\beta/2$' if relative else '')
style_axis(axes[0, 1], ylabel)
style_axis(axes[1, 1], 'CTHYB $-$ ED')
axes[0, 1].legend(fontsize=8, ncol=2, frameon=False)
names = ',  '.join(f'$O_{i}$ = {combination_name(v)}' for i, v in enumerate(conserved_vectors))
axes[0, 1].set_title('Conserved density combinations\n' + names, color=INK, loc='left', fontsize=9)

p = ed['params']
fig.suptitle(fr"$\beta$={p['beta']}, U={p['U']}, J={p['J']}, V={p['V']}, $\omega_0$={p['omega_0']}, "
             fr"g=({p['g_orb0']}, {p['g_orb1']});  CTHYB: lang_firsov={lang_firsov}, {n_cycles} cycles, sign {np.real(average_sign):.3f}",
             color=INK, fontsize=10)

out = args.out or os.path.splitext(args.cthyb_file)[0] + '_vs_ed.png'
fig.savefig(out, dpi=150)

# Second figure: orbital-resolved <n_a(tau) n_b(0)> from the stochastic vertices (the
# coupling-derivative estimator). Unlike the kink estimator this carries its own equal-time value,
# so it is compared to ED directly. Points the estimator had to mask (negligible coupling) are
# stored as exactly 0 and left out of the line.
if dyn_corr is not None:
    density_types = [t for t in range(len(dyn_pairs)) if dyn_pairs[t][0] >= 0 and dyn_pairs[t][1] >= 0]
    if density_types:
        fig2, ax2 = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]), constrained_layout=True)
        for k, t in enumerate(density_types[:len(SERIES_COLORS)]):
            a, b = dyn_pairs[t]
            name = f"n({labels[a][0]},{labels[a][1]})-n({labels[b][0]},{labels[b][1]})"
            ed_curve = ed['chi'][a, b]
            cthyb_curve = np.where(dyn_corr[t] == 0.0, np.nan, dyn_corr[t])
            color = SERIES_COLORS[k]
            ax2[0].plot(ed['tau'], ed_curve, color=color, linewidth=2, label=f'ED  {name}')
            ax2[0].plot(dyn_tau, cthyb_curve, color=color, linewidth=1, linestyle=(0, (4, 2)), label=f'CTHYB  {name}')
            residual = cthyb_curve - np.interp(dyn_tau, ed['tau'], ed_curve)
            ax2[1].plot(dyn_tau, residual, color=color, linewidth=1)
            summary.append(f"chi {name}: max |CTHYB - ED| = {np.nanmax(np.abs(residual)):.4f}")
        style_axis(ax2[0], r'$\langle n_a(\tau) n_b(0)\rangle$')
        style_axis(ax2[1], 'CTHYB $-$ ED')
        ax2[0].legend(fontsize=8, frameon=False)
        ax2[0].set_title('From the stochastic vertices (coupling derivative)', color=INK, loc='left')
        out2 = os.path.splitext(out)[0] + '_chi_ab.png'
        fig2.savefig(out2, dpi=150)
        print(f"Saved {out2}")

print('\n'.join(summary))
print(f"Saved {out}")
