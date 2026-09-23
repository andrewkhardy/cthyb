# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# Independent-chain diagnostics, output of run_chains.sh. One row per cell.
#
# Left: <n> of every chain, one column per series. Right: <n> against <k_dyn>, the chain's
# mean number of stochastic vertices. Series that sample the same distribution fall on one
# cluster; a solver or move set that samples a different one shows up as its own cluster,
# in either panel. The printout gives each series' mean and its standard error over chains.
#
# Knobs hardcoded below so this pastes straight into a notebook.
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

# ---------------------------------------------------------------------------------- knobs
CHAIN_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin/chains4"
SEED_STEP = 2
# (title, beta, filling, jperp, szsz, [(series label, solver, lang_firsov, first seed, n chains)])
# -- as in run_chains.sh. lang_firsov is None for CTSEG (no such tag in its file names).
# Round 3 (chains3), all beta = 100: A full S.S n = 0.5 [CTSEG 1000x8, CTHYB old 1000x12,
# local 1100x12, flip 1200x12], B Sz.Sz only [CTSEG 2000x8, CTHYB 2000x16], C Jperp only
# [CTSEG 3000x8, CTHYB old 3000x12, local 3100x8].
CELLS = [
    ("Sz.Sz only, beta = 100", 100.0, 0.75, 0, 1,
     [("CTSEG", "ctseg", None, 1000, 8), ("CTHYB ref", "cthyb", True, 1000, 12),
      ("CTHYB dyn_n_l 150", "cthyb", True, 1100, 12), ("CTHYB lc 100 + double", "cthyb", True, 1200, 12)]),
    ("Sz.Sz only, beta = 30", 30.0, 0.75, 0, 1,
     [("CTSEG", "ctseg", None, 2000, 6), ("CTHYB ref", "cthyb", True, 2000, 6)]),
    ("Sz.Sz only, beta = 10", 10.0, 0.75, 0, 1,
     [("CTSEG", "ctseg", None, 3000, 6), ("CTHYB ref", "cthyb", True, 3000, 6)]),
    ("full S.S, beta = 100, mu(n = 0.75)", 100.0, 0.75, 1, 1,
     [("CTSEG", "ctseg", None, 4000, 8), ("CTHYB new moves", "cthyb", True, 4000, 20)]),
]
J = 1.0
# -----------------------------------------------------------------------------------------

COLOR = {"CTSEG": "#1f6feb", "CTHYB ref": "#e8710a", "CTHYB dyn_n_l 150": "#2a9d3f",
         "CTHYB lc 100 + double": "#9b5de5", "CTHYB new moves": "#c1121f"}

fig, axes = plt.subplots(len(CELLS), 2, figsize=(11, 3.6 * len(CELLS)), squeeze=False)
for row, (title, beta, filling, jperp, szsz, series) in enumerate(CELLS):
    ax_n, ax_k = axes[row]
    for col, (label, solver, lang_firsov, seed0, count) in enumerate(series):
        dens, order, signs = [], [], []
        for seed in range(seed0, seed0 + SEED_STEP * count, SEED_STEP):
            name = (f"spin_spin_{solver}_b-{beta:g}_n-{filling:g}_J-{J:g}_jperp-{jperp:g}_szsz-{szsz:g}"
                    + (f"_lf-{lang_firsov}" if lang_firsov is not None else "") + f"_seed-{seed}.h5")
            path = os.path.join(CHAIN_DIR, name)
            if not os.path.exists(path):
                continue
            with HDFArchive(path, "r") as A:
                dens.append(np.mean(A["density"]))
                signs.append(float(A["average_sign"]))
                h = np.asarray(A["pert_order_dyn"], dtype=float) if "pert_order_dyn" in A else np.array([1.0])
            order.append((np.arange(len(h)) * h).sum() / h.sum())
        if not dens:
            print(f"[{title}] {label}: no chain files")
            continue
        dens, order = np.array(dens), np.array(order)

        jitter = col + 0.08 * np.random.default_rng(0).standard_normal(dens.size)
        ax_n.plot(jitter, dens, "o", color=COLOR[label], alpha=0.7, label=label)
        ax_k.plot(order, dens, "o", color=COLOR[label], alpha=0.7, label=label)
        sem = lambda x: x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else float("nan")
        print(f"[{title}] {label:12s} {dens.size:2d} chains  <n> = {dens.mean():.4f} +- {sem(dens):.4f}  "
              f"(chain spread {dens.std(ddof=1):.4f})   <k_dyn> = {order.mean():6.3f} +- {sem(order):.3f}"
              f"   sign >= {min(signs):.3f}")

    ax_n.set_xticks(range(len(series)), [s[0] for s in series], fontsize=8)
    ax_n.set_ylabel(r"$\langle n\rangle$ per spin-orbital")
    ax_k.set_xlabel(r"$\langle k_\mathrm{dyn}\rangle$ of the chain")
    ax_k.set_ylabel(r"$\langle n\rangle$")
    for ax in (ax_n, ax_k):
        ax.set_title(title)
        ax.legend(fontsize=7)

fig.tight_layout()
plt.show()
