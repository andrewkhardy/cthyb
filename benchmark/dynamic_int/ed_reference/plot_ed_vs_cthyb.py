# Compare a CTHYB run (cthyb_kanamori_phonon.py) against the ED reference (ed_kanamori_phonon.py)
# for the same model. Two figures:
#   green_and_conserved: G_a(tau) per spin-orbital, and <O_i(tau) O_j(0)> for the conserved density
#                        combinations CTHYB measured (Q_conserved_tau), each with CTHYB - ED below.
#   chi_ab:              orbital-resolved <n_a(tau) n_b(0)> from the stochastic vertices (the
#                        coupling-derivative estimator, dyn_vertex_corr_tau), against ED.
#
# Runs either way:
#   - in Jupyter/IPython, just run the cells; edit the paths in the CONFIG block below
#   - from the shell: python plot_ed_vs_cthyb.py <ed_file> <cthyb_file> [--g_bins 100] [--out fig.png]

import os
import sys
import numpy as np
from h5 import HDFArchive
from scipy.interpolate import CubicSpline
import triqs.gfs  # registers Gf / BlockGf with h5


def _ipython():
    """The IPython/Jupyter shell, or None when running as a plain script."""
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


# Under IPython the default backend is often non-interactive (Agg), which makes plt.show() a
# no-op with a warning: switch to an interactive one so the figures actually appear.
_SHELL = _ipython()
if _SHELL is not None:
    for _backend in ('widget', 'inline'):  # widget (ipympl) is zoomable; inline always works
        try:
            _SHELL.run_line_magic('matplotlib', _backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt  # imported after the magic, so it picks up that backend

# ---------------------------------------------------------------- CONFIG (edit for interactive use)
DATA_DIR = '/home/andrewhardy/Documents/Data/CTHYB_Data/ed_reference'
TAG = 'beta-10.0_U-2.0_J-0.3_V-0.7_eb-0.0_w0-1.0_g-0.7-0.3_mu-half'
ED_FILE = f'{DATA_DIR}/ed_{TAG}_nph-24.h5'
CTHYB_FILE = f'{DATA_DIR}/cthyb_{TAG}_lf-True_nc-1000000.h5'
G_BINS = 100     # CTHYB G_tau is a per-bin histogram: compare averages over this many tau bins
OUT = None       # None: next to the CTHYB file, as <cthyb_file>_vs_ed.png
# ----------------------------------------------------------------------------------------------

# Categorical slots 1-4, fixed order; series are also told apart by legend, and source by line style
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'


def in_notebook():
    """argparse would choke on the -f <kernel.json> Jupyter passes, so only use it outside one."""
    return _SHELL is not None


def read_inputs(ed_file, cthyb_file):
    with HDFArchive(ed_file, 'r') as A:
        ed = {key: A[key] for key in ['tau', 'chi', 'G', 'labels', 'params']}
    with HDFArchive(cthyb_file, 'r') as A:
        keys = list(A.keys())
        cthyb = dict(G_tau=A['G_tau'], Q_conserved_tau=A['Q_conserved_tau'],
                     conserved_vectors=A['conserved_vectors'], equal_time_added=A['equal_time_added'],
                     lang_firsov=A['lang_firsov'], average_sign=A['average_sign'], n_cycles=A['n_cycles'],
                     params=A['params'],
                     dyn_pairs=A['dyn_vertex_pairs'] if 'dyn_vertex_pairs' in keys else None,
                     dyn_corr=A['dyn_vertex_corr'] if 'dyn_vertex_corr' in keys else None,
                     dyn_tau=A['dyn_vertex_tau'] if 'dyn_vertex_tau' in keys else None)
    for key, value in ed['params'].items():
        if not np.isclose(value, cthyb['params'][key]):
            raise ValueError(f"ED and CTHYB files are for different models: {key} = {value} vs {cthyb['params'][key]}")
    return ed, cthyb


def bin_average(values, n_bins):
    return np.array([chunk.mean() for chunk in np.array_split(values, n_bins)])


def ed_on(tau, ed_tau, curve):
    """An ED curve on the (finer) CTHYB mesh.

    Cubic, not linear: G and chi have a kink at tau = 0, so linear interpolation across the
    ED mesh spacing errs by ~1e-4 in the first and last intervals - comparable to the residual
    being plotted, and localised exactly at the tails where it would be read as a systematic.
    See diagnose_residual.py, part B.
    """
    return CubicSpline(ed_tau, curve)(tau)


def orbital_name(label):
    return f"n({label[0]},{label[1]})"


def combination_name(v, labels):
    terms = [(f"{c:g}" if abs(c - 1) > 1e-12 else "") + orbital_name(label) for c, label in zip(v, labels) if abs(c) > 1e-12]
    return " + ".join(terms)


def style_axis(ax, ylabel, beta):
    ax.set_xlabel(r'$\tau$', color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlim(0, beta)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_green_and_conserved(ed, cthyb, g_bins=G_BINS, summary=None):
    """G_a(tau) and the conserved-combination correlators, CTHYB against ED."""
    summary = summary if summary is not None else []
    beta = ed['params']['beta']
    labels = [tuple(label.split(',')) for label in ed['labels']]
    G_tau, Q_conserved_tau = cthyb['G_tau'], cthyb['Q_conserved_tau']

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]),
                             constrained_layout=True)

    for a, (s, o) in enumerate(labels):
        tau_c = np.linspace(0, beta, len(G_tau[s].mesh))
        # Same bins for both: CTHYB's histogram, and the (smooth) ED curve interpolated onto its mesh
        tau_b = bin_average(tau_c, g_bins)
        g_cthyb = bin_average(G_tau[s].data[:, int(o), int(o)].real, g_bins)
        g_ed = bin_average(ed_on(tau_c, ed['tau'], ed['G'][a]), g_bins)
        color = SERIES_COLORS[a]
        axes[0, 0].plot(ed['tau'], ed['G'][a], color=color, linewidth=2, label=f'ED  {orbital_name((s, o))}')
        axes[0, 0].plot(tau_b, g_cthyb, color=color, linewidth=1, linestyle=(0, (4, 2)), label=f'CTHYB  {orbital_name((s, o))}')
        residual = g_cthyb - g_ed
        axes[1, 0].plot(tau_b, residual, color=color, linewidth=1)
        summary.append(f"G({s},{o}): max |CTHYB - ED| over {g_bins} bins = {np.abs(residual).max():.4f}")
    style_axis(axes[0, 0], r'$G_a(\tau)$', beta)
    style_axis(axes[1, 0], 'CTHYB $-$ ED', beta)
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    axes[0, 0].set_title(f'Green function (CTHYB averaged over {g_bins} bins)', color=INK, loc='left')

    # Without the density matrix, Q_conserved_tau lacks the equal-time constant <O_i O_j>:
    # compare shapes relative to beta/2 instead
    relative = not cthyb['equal_time_added']
    tau_q = np.linspace(0, beta, len(Q_conserved_tau.mesh))
    mid_ed, mid_q = np.argmin(np.abs(ed['tau'] - beta / 2)), np.argmin(np.abs(tau_q - beta / 2))
    vectors = cthyb['conserved_vectors']
    for k, (i, j) in enumerate([(i, j) for i in range(len(vectors)) for j in range(i, len(vectors))]):
        ed_curve = np.einsum('a,abt,b->t', vectors[i], ed['chi'], vectors[j])
        q_curve = Q_conserved_tau.data[:, i, j].real
        if relative:
            ed_curve, q_curve = ed_curve - ed_curve[mid_ed], q_curve - q_curve[mid_q]
        color = SERIES_COLORS[k % len(SERIES_COLORS)]
        axes[0, 1].plot(ed['tau'], ed_curve, color=color, linewidth=2, label=fr'ED  $\langle O_{i} O_{j}\rangle$')
        axes[0, 1].plot(tau_q, q_curve, color=color, linewidth=1, linestyle=(0, (4, 2)), label=fr'CTHYB  $\langle O_{i} O_{j}\rangle$')
        residual = q_curve - ed_on(tau_q, ed['tau'], ed_curve)
        axes[1, 1].plot(tau_q, residual, color=color, linewidth=1)
        summary.append(f"<O_{i} O_{j}>: max |CTHYB - ED| = {np.abs(residual).max():.4f}")
    ylabel = r'$\langle O_i(\tau) O_j(0)\rangle$' + (r' $-$ value at $\beta/2$' if relative else '')
    style_axis(axes[0, 1], ylabel, beta)
    style_axis(axes[1, 1], 'CTHYB $-$ ED', beta)
    axes[0, 1].legend(fontsize=8, ncol=2, frameon=False)
    names = ',  '.join(f'$O_{i}$ = {combination_name(v, labels)}' for i, v in enumerate(vectors))
    axes[0, 1].set_title('Conserved density combinations\n' + names, color=INK, loc='left', fontsize=9)

    p = ed['params']
    fig.suptitle(fr"$\beta$={p['beta']}, U={p['U']}, J={p['J']}, V={p['V']}, $\omega_0$={p['omega_0']}, "
                 fr"g=({p['g_orb0']}, {p['g_orb1']});  CTHYB: lang_firsov={cthyb['lang_firsov']}, "
                 fr"{cthyb['n_cycles']} cycles, sign {np.real(cthyb['average_sign']):.3f}", color=INK, fontsize=10)
    return fig, summary


def plot_chi_ab(ed, cthyb, summary=None):
    """Orbital-resolved <n_a(tau) n_b(0)> from the stochastic vertices, against ED.

    Unlike the kink estimator this carries its own equal-time value, so it is compared to ED
    directly. Points the estimator had to mask (negligible coupling) are stored as exactly 0 and
    left out of the line. Returns None when the run had no stochastic density vertices.
    """
    summary = summary if summary is not None else []
    dyn_pairs, dyn_corr, dyn_tau = cthyb['dyn_pairs'], cthyb['dyn_corr'], cthyb['dyn_tau']
    if dyn_corr is None:
        return None, summary
    density_types = [t for t in range(len(dyn_pairs)) if dyn_pairs[t][0] >= 0 and dyn_pairs[t][1] >= 0]
    if not density_types:
        return None, summary

    beta = ed['params']['beta']
    labels = [tuple(label.split(',')) for label in ed['labels']]
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]),
                           constrained_layout=True)
    for k, t in enumerate(density_types[:len(SERIES_COLORS)]):
        a, b = dyn_pairs[t]
        name = f"{orbital_name(labels[a])}-{orbital_name(labels[b])}"
        ed_curve = ed['chi'][a, b]
        cthyb_curve = np.where(dyn_corr[t] == 0.0, np.nan, dyn_corr[t])
        color = SERIES_COLORS[k]
        ax[0].plot(ed['tau'], ed_curve, color=color, linewidth=2, label=f'ED  {name}')
        ax[0].plot(dyn_tau, cthyb_curve, color=color, linewidth=1, linestyle=(0, (4, 2)), label=f'CTHYB  {name}')
        residual = cthyb_curve - ed_on(dyn_tau, ed['tau'], ed_curve)
        ax[1].plot(dyn_tau, residual, color=color, linewidth=1)
        summary.append(f"chi {name}: max |CTHYB - ED| = {np.nanmax(np.abs(residual)):.4f}")
    style_axis(ax[0], r'$\langle n_a(\tau) n_b(0)\rangle$', beta)
    style_axis(ax[1], 'CTHYB $-$ ED', beta)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title('From the stochastic vertices (coupling derivative)', color=INK, loc='left')
    return fig, summary


# Running the file (shell or "run all cells") builds both figures; importing it does not.
if __name__ == '__main__':
    ed_file, cthyb_file, g_bins, out = ED_FILE, CTHYB_FILE, G_BINS, OUT
    if not in_notebook() and len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='Plot CTHYB against the ED reference.')
        parser.add_argument('ed_file', nargs='?', default=ED_FILE)
        parser.add_argument('cthyb_file', nargs='?', default=CTHYB_FILE)
        parser.add_argument('--g_bins', type=int, default=G_BINS)
        parser.add_argument('--out', default=OUT, help='Output image (default: next to cthyb_file)')
        args = parser.parse_args()
        ed_file, cthyb_file, g_bins, out = args.ed_file, args.cthyb_file, args.g_bins, args.out

    ed, cthyb = read_inputs(ed_file, cthyb_file)
    summary = []
    fig_main, summary = plot_green_and_conserved(ed, cthyb, g_bins, summary)
    fig_chi, summary = plot_chi_ab(ed, cthyb, summary)

    out = out or os.path.splitext(cthyb_file)[0] + '_vs_ed.png'
    fig_main.savefig(out, dpi=150)
    print(f"Saved {out}")
    if fig_chi is not None:
        out_chi = os.path.splitext(out)[0] + '_chi_ab.png'
        fig_chi.savefig(out_chi, dpi=150)
        print(f"Saved {out_chi}")
    print('\n'.join(summary))
    if in_notebook():
        plt.show()
