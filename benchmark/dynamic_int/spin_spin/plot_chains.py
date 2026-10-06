# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# Independent-chain diagnostics for the output of run_chains.sh, one row per cell. Left: <n>
# of every chain. Middle: <n> against the chain's <k_dyn>; a series sampling a different
# distribution shows up as its own cluster. Right: <Sz(tau)Sz(0)> averaged over chains, solid
# the direct estimator (CTHYB O_tau, CTSEG nn_tau), dashed CTHYB's Legendre kink estimator.
# Prints each series' mean and standard error over chains.
#
#   python plot_chains.py      (knobs below)
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

# ---------------------------------------------------------------------------------- knobs
CHAIN_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin/chains6"
SEED_STEP = 2
# (title, beta, filling, jperp, szsz, [(series label, solver, lang_firsov, first seed, n chains)]),
# as in run_chains.sh; lang_firsov is None for CTSEG, whose file names have no lf tag.
CELLS = [
    ("full S.S, beta = 100, mu(n = 0.75)", 100.0, 0.75, 1, 1,
     [("CTSEG", "ctseg", None, 5000, 24), ("CTHYB lc 2000", "cthyb", True, 5200, 72)]),
]
J = 1.0
# -----------------------------------------------------------------------------------------

COLOR = {"CTSEG": "#1f6feb", "CTHYB lc 2000": "#c1121f"}

fig, axes = plt.subplots(len(CELLS), 3, figsize=(16.5, 3.6 * len(CELLS)), squeeze=False)
for row, (title, beta, filling, jperp, szsz, series) in enumerate(CELLS):
    ax_n, ax_k, ax_c = axes[row]
    for col, (label, solver, lang_firsov, seed0, count) in enumerate(series):
        dens, order, signs, mtimes = [], [], [], []
        corr, corr_alt, tau_corr = [], [], None
        for seed in range(seed0, seed0 + SEED_STEP * count, SEED_STEP):
            name = (f"spin_spin_{solver}_b-{beta:g}_n-{filling:g}_J-{J:g}_jperp-{jperp:g}_szsz-{szsz:g}"
                    + (f"_lf-{lang_firsov}" if lang_firsov is not None else "") + f"_seed-{seed}.h5")
            path = os.path.join(CHAIN_DIR, name)
            if not os.path.exists(path):
                continue
            mtimes.append(os.path.getmtime(path))
            with HDFArchive(path, "r") as A:
                dens.append(np.mean(A["density"]))
                signs.append(float(A["average_sign"]))
                h = np.asarray(A["pert_order_dyn"], dtype=float) if "pert_order_dyn" in A else np.array([1.0])
                tau_corr = np.asarray(A["tau_corr"])
                corr.append(np.asarray(A["corr"]))
                if "corr_alt" in A:
                    corr_alt.append(np.asarray(A["corr_alt"]))
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
              f"   sign >= {min(signs):.3f}   files written "
              # when the files were written, to spot stale ones from an earlier round
              f"{time.strftime('%m-%d %H:%M', time.localtime(min(mtimes)))} .. "
              f"{time.strftime('%H:%M', time.localtime(max(mtimes)))}")

        # The kink estimator is on the uniform bosonic mesh on [0, beta]
        corr = np.array(corr)
        ax_c.plot(tau_corr / beta, corr.mean(axis=0), color=COLOR[label], label=label)
        mid = np.array([np.interp(beta / 2, tau_corr, c) for c in corr])
        line = f"{'':15s}<SzSz>(beta/2): direct {mid.mean():.5f} +- {sem(mid):.5f}"
        if corr_alt:
            corr_alt = np.array(corr_alt)
            tau_alt = np.linspace(0.0, beta, corr_alt.shape[1])
            ax_c.plot(tau_alt / beta, corr_alt.mean(axis=0), color=COLOR[label], linestyle="--",
                      label=f"{label} (kink)")
            mid_alt = np.array([np.interp(beta / 2, tau_alt, c) for c in corr_alt])
            line += f"   kink {mid_alt.mean():.5f} +- {sem(mid_alt):.5f}"
        print(line)

    ax_n.set_xticks(range(len(series)), [s[0] for s in series], fontsize=8)
    ax_n.set_ylabel(r"$\langle n\rangle$ per spin-orbital")
    ax_k.set_xlabel(r"$\langle k_\mathrm{dyn}\rangle$ of the chain")
    ax_k.set_ylabel(r"$\langle n\rangle$")
    ax_c.set_xlabel(r"$\tau/\beta$")
    ax_c.set_ylabel(r"$\langle S_z(\tau)S_z(0)\rangle$, mean over chains")
    for ax in (ax_n, ax_k, ax_c):
        ax.set_title(title)
        ax.legend(fontsize=7)

fig.tight_layout()
plt.show()
