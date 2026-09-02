# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Analysis/plotting for kanamori_dyn_spinflip.py's output (also reads kanamori_dyn.py's,
# since the h5 layout is the same modulo the extra g_sf entry). Pass one or more result
# files; each is overlaid on the same axes so e.g. a lang_firsov=True run and its
# forced-stochastic counterpart, or g_sf=0 vs g_sf!=0, can be compared directly.
#
#   python plot_kanamori_dyn_spinflip.py run1.h5 [run2.h5 ...] --out comparison.png
#
# Three panels:
#   1. G(tau) per block -- the physical result.
#   2. G(tau) - G(beta-tau), the particle-hole asymmetry. With the mu correction applied
#      (see kanamori_dyn.py and doc/notes/dynamical_interactions.tex Sec. 5) this should
#      sit at noise level; a systematic offset means the model is off half filling.
#   3. The dynamical perturbation-order histogram -- the diagnostic that matters for the
#      spin-flip channel specifically, since those vertices are the only ones sampled
#      stochastically when lang_firsov=True. A histogram concentrated entirely at order 0
#      means the stochastic channel is effectively never being inserted (check g_sf).

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from h5 import HDFArchive
from triqs.gfs import *

plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rc('font', size=11)

parser = argparse.ArgumentParser(description="Plot/compare kanamori_dyn_spinflip.py results.")
parser.add_argument('files', nargs='+', help='One or more result h5 files')
parser.add_argument('--out', default='kanamori_dyn_spinflip_comparison.png', help='Output image path')
parser.add_argument('--block', default=None, help='Only plot this block (default: all blocks)')
args = parser.parse_args()


def load(filename):
    with HDFArchive(filename, "r") as A:
        d = {k: A[k] for k in ('G_tau', 'average_sign', 'beta', 'U', 'J', 'g', 'omega_0',
                               'lang_firsov', 'mu', 'mu_bare', 'n_orb') if k in A}
        d['g_sf'] = A['g_sf'] if 'g_sf' in A else 0.0
        d['pert_dyn'] = np.asarray(A['perturbation_order_dynamical']) if 'perturbation_order_dynamical' in A else None
    return d


runs = [(f, load(f)) for f in args.files]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

for idx, (filename, d) in enumerate(runs):
    label_base = f"g={d.get('g')}, g_sf={d['g_sf']}, LF={d.get('lang_firsov')}"
    colour = f"C{idx}"

    print(f"\n=== {filename} ===")
    print(f"  U={d.get('U')}, J={d.get('J')}, beta={d.get('beta')}, omega_0={d.get('omega_0')}, "
          f"g={d.get('g')}, g_sf={d['g_sf']}, lang_firsov={d.get('lang_firsov')}")
    print(f"  average sign = {d.get('average_sign')}")
    print(f"  mu_bare = {d.get('mu_bare')}, corrected mu = {d.get('mu')}")

    for b_idx, (name, g) in enumerate(d['G_tau']):
        if args.block is not None and name != args.block:
            continue
        tau = np.array([float(t) for t in g.mesh])
        for orb in range(g.data.shape[1]):
            gd = g.data[:, orb, orb].real
            style = ['-', '--', ':', '-.'][b_idx % 4]
            axes[0].plot(tau, gd, style, color=colour, lw=1.2, alpha=0.8,
                         label=f"{label_base} [{name},{orb}]" if (b_idx == 0 and orb == 0) else None)
            asym = gd - gd[::-1]
            axes[1].plot(tau, asym, style, color=colour, lw=1.0, alpha=0.8)
            print(f"  {name}[{orb},{orb}]: G(0)={gd[0]:+.4f}, G(beta)={gd[-1]:+.4f}, "
                  f"n={1.0 + gd[0]:.4f}, max|G(tau)-G(beta-tau)|={np.max(np.abs(asym)):.4f}")

    if d['pert_dyn'] is not None:
        counts = d['pert_dyn']
        total = counts.sum()
        support = np.nonzero(counts)[0]
        hi = support.max() if support.size else 0
        mean_order = float((np.arange(counts.size) * counts).sum() / total) if total > 0 else 0.0
        axes[2].plot(np.arange(hi + 2), counts[:hi + 2] / max(total, 1), 'o-', color=colour,
                     lw=1.2, ms=3, label=label_base)
        print(f"  dynamical perturbation order: mean={mean_order:.3f}, max occupied={hi}, "
              f"fraction at order 0 = {counts[0] / max(total, 1):.4f}")
    else:
        print("  dynamical perturbation order: NOT MEASURED "
              "(no vertex was routed to the stochastic path -- expected only if g_sf=0)")

axes[0].set_xlabel(r"$\tau$")
axes[0].set_ylabel(r"$G(\tau)$")
axes[0].set_title(r"$G(\tau)$ (diagonal components)")
axes[0].legend(fontsize=7)

axes[1].axhline(0.0, color='k', lw=0.6)
axes[1].set_xlabel(r"$\tau$")
axes[1].set_ylabel(r"$G(\tau)-G(\beta-\tau)$")
axes[1].set_title("Particle-hole asymmetry")

axes[2].set_xlabel("dynamical perturbation order")
axes[2].set_ylabel("fraction of measures")
axes[2].set_title("Stochastic (spin-flip) vertex insertions")
axes[2].set_yscale('log')
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig(args.out)
print(f"\nSaved {args.out}")
