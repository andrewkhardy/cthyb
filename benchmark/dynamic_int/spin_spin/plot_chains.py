# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# Independent-chain ergodicity diagnostic, output of run_chains.sh.
#
# Left: density of every chain, one column per solver, with the 96-rank production value
# as a line. Right: each chain's density against its weight in the no-moment mode,
# P(k_dyn < K_SPLIT). If the chains mix, the points form one tight cluster per solver. If
# they do not, they spread along a line -- the density is set by which mode a chain sat in
# -- and the two solvers' clusters sitting at different points on that same line means
# they sample the same distribution with different mixing, not different Hamiltonians.
#
# Knobs hardcoded below so this pastes straight into a notebook.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import io

# ---------------------------------------------------------------------------------- knobs
CHAIN_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin/chains2"
PROD_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin"
# (beta, filling, [(series label, first seed, number of chains), ...]) -- as in run_chains.sh.
# Round 1 (CHAIN_DIR .../chains, SEED_STEP 1): CTSEG and CTHYB at seeds 1000 x 24
# (beta = 100, n = 0.75), 2000 x 12 (n = 0.5), 3000 x 12 (beta = 10) -- but seeds 2k, 2k+1
# there are the same chain (RandMT forces odd).
CELLS = [(100.0, 0.75, [("CTSEG", 1000, 32), ("CTHYB old moves", 1000, 16), ("CTHYB new moves", 1100, 24)]),
         (100.0, 0.5, [("CTSEG", 2000, 16), ("CTHYB new moves", 2100, 8)])]
SEED_STEP = 2
K_SPLIT = 4            # k_dyn below this counts as the no-moment mode
TAG = "J-1_jperp-1_szsz-1"
# -----------------------------------------------------------------------------------------

# series label -> (solver, filename suffix)
SERIES = {"CTSEG": ("ctseg", ""), "CTHYB old moves": ("cthyb", "_lf-True"), "CTHYB new moves": ("cthyb", "_lf-True")}
COLOR = {"CTSEG": "#1f6feb", "CTHYB old moves": "#e8710a", "CTHYB new moves": "#2a9d3f"}


def low_mode_weight(run):
    h = np.asarray(run["pert_order_dyn"], dtype=float).ravel()
    return h[:K_SPLIT].sum() / h.sum()


fig, axes = plt.subplots(len(CELLS), 2, figsize=(11, 3.6 * len(CELLS)), squeeze=False)
for row, (beta, filling, series) in enumerate(CELLS):
    ax_n, ax_w = axes[row]
    for col, (label, seed0, count) in enumerate(series):
        solver, suffix = SERIES[label]
        dens, weight = [], []
        for seed in range(seed0, seed0 + SEED_STEP * count, SEED_STEP):
            run = io.load(io.output_file(CHAIN_DIR, "spin_spin", solver, beta, filling,
                                         tag=f"{TAG}{suffix}_seed-{seed}"))
            if run is None:
                continue
            dens.append(np.mean(run["density"]))
            weight.append(low_mode_weight(run))
        dens, weight = np.array(dens), np.array(weight)
        if dens.size == 0:
            print(f"[beta={beta:g} n={filling:g}] {label}: no chain files")
            continue

        jitter = col + 0.08 * np.random.default_rng(0).standard_normal(dens.size)
        ax_n.plot(jitter, dens, "o", color=COLOR[label], alpha=0.7, label=label)
        prod = io.load(io.output_file(PROD_DIR, "spin_spin", solver, beta, filling, tag=f"{TAG}{suffix}"))
        if prod is not None:
            ax_n.hlines(np.mean(prod["density"]), col - 0.3, col + 0.3, color=COLOR[label],
                        linewidth=2, label=f"{label} production (96 ranks)")
        ax_w.plot(weight, dens, "o", color=COLOR[label], alpha=0.7, label=label)

        print(f"[beta={beta:g} n={filling:g}] {label}: {dens.size} chains  "
              f"<n> = {dens.mean():.4f}  chain spread (std) = {dens.std(ddof=1):.4f}  "
              f"std of mean = {dens.std(ddof=1) / np.sqrt(dens.size):.4f}  "
              f"range = [{dens.min():.4f}, {dens.max():.4f}]  "
              f"P(k_dyn<{K_SPLIT}) = {weight.mean():.3f} +- {weight.std(ddof=1):.3f}")

    ax_n.set_xticks(range(len(series)), [label for label, _, _ in series], fontsize=8)
    ax_n.set_ylabel(r"$\langle n\rangle$ per spin-orbital, one point per chain")
    ax_n.set_title(r"$\beta$" + f"={beta:g}, n={filling:g}")
    ax_n.legend(fontsize=7)
    ax_w.set_xlabel(f"P(k_dyn < {K_SPLIT})  (no-moment mode weight)")
    ax_w.set_ylabel(r"$\langle n\rangle$")
    ax_w.set_title(r"$\beta$" + f"={beta:g}, n={filling:g}")
    ax_w.legend(fontsize=7)

fig.tight_layout()
plt.show()
