# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# nn_tau = <n_a(tau) n_b(0)> from run_cthyb.py --measure_nn_tau True, against ED's chi[a, b], which is
# exact up to the phonon truncation. First figure: the residual of every pair, a and b in model.labels
# order. Second: the conserved combinations <N_s(tau) N_s'(0)>, where the occupation-kink estimator
# (measure_D0_corr, Q_conserved_tau) measures the same correlator. Knobs below; runs cell by cell in Jupyter.
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gfs import Gf  # noqa: F401
# Unused, but it registers the h5 reader for Q_conserved_tau.

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon"
TAG = "beta-10.0_U-2.0_J-0.3_V-0.7_eb-0.0_w0-1.0_g-0.5-0.5_mu-half"  # model.Model.tag()
N_PH = 24
LABELS = ["up,0", "up,1", "dn,0", "dn,1"]  # model.labels
SAVE_AS = None  # e.g. "nn_tau_vs_ed.pdf"; the second figure gets a "_conserved" suffix
# -----------------------------------------------------------------------------------------

INK, MUTED = "#0b0b0b", "#8a8984"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2


def load(path):
    with HDFArchive(path, "r") as A:
        return {k: A[k] for k in A.keys()}


def collect():
    """The ED reference and the most recent CTHYB run of TAG that measured nn_tau."""
    ed = load(os.path.join(DATA_DIR, f"ed_{TAG}_nph-{N_PH}.h5"))
    paths = sorted(glob.glob(os.path.join(DATA_DIR, f"cthyb_{TAG}_lf-*_nntau*.h5")), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"no cthyb_{TAG}_lf-*_nntau*.h5 in {DATA_DIR}: run_kanamori_phonon.sh nntau writes it")
    print(f"run: {os.path.basename(paths[-1])}")
    return ed, load(paths[-1])


def style(ax):
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def plot_pairs(ed, run):
    beta = ed["params"]["beta"]
    tau, nn = run["nn_tau_tau"], run["nn_tau"]
    chi = np.array([[np.interp(tau, ed["tau"], ed["chi"][a, b]) for b in range(len(LABELS))] for a in range(len(LABELS))])
    fig, axes = plt.subplots(len(LABELS), len(LABELS), sharex=True, sharey=True, figsize=(10, 8))
    print("max |nn_tau - ED|, row a = n_a(tau), column b = n_b(0):")
    for a, row in enumerate(axes):
        for b, ax in enumerate(row):
            res = nn[a, b] - chi[a, b]
            ax.plot(tau / beta, res, color=SERIES[0], linewidth=1.5)
            ax.set_title(f"<n_{LABELS[a]}(τ) n_{LABELS[b]}(0)>   max|Δ| {np.abs(res).max():.1e}", fontsize=7, color=INK)
            style(ax)
        print("  " + "  ".join(f"{np.abs(nn[a, b] - chi[a, b]).max():.2e}" for b in range(len(LABELS))))
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau/\beta$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("CTHYB − ED", fontsize=8)
    fig.suptitle(f"nn_tau against ED, {run['n_cycles']} cycles per rank, solve {run['solve_seconds']:.0f} s", fontsize=9)
    fig.tight_layout()
    return fig


def plot_conserved(ed, run):
    """<O_i(tau) O_j(0)> of the conserved combinations, by occupation kinks and from nn_tau, against ED."""
    beta = ed["params"]["beta"]
    tau, nn, v = run["nn_tau_tau"], run["nn_tau"], run["conserved_vectors"]
    chi = np.array([[np.interp(tau, ed["tau"], ed["chi"][a, b]) for b in range(len(LABELS))] for a in range(len(LABELS))])
    exact, sweep = np.einsum("ia,jb,abt->ijt", v, v, chi), np.einsum("ia,jb,abt->ijt", v, v, nn)
    Q = run["Q_conserved_tau"]
    kinks = np.array([[np.interp(tau, [float(t) for t in Q.mesh], Q.data[:, i, j].real) for j in range(len(v))] for i in range(len(v))])
    names = [f"O_{i}" for i in range(len(v))]
    print("conserved combinations: " + ", ".join(f"O_{i} = {op}" for i, op in enumerate(run["conserved_operators"])))
    fig, axes = plt.subplots(len(v), len(v), sharex=True, sharey=True, figsize=(8, 5.5), squeeze=False)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            for estimate, label, color in ((kinks, "occupation kinks (Q_conserved_tau)", SERIES[0]), (sweep, "sweep (nn_tau)", SERIES[1])):
                res = estimate[i, j] - exact[i, j]
                ax.plot(tau / beta, res, color=color, linewidth=1.5, label=label)
                print(f"  <{names[i]}(tau) {names[j]}(0)> {label}: max |Δ| {np.abs(res).max():.2e}, rms {np.sqrt((res**2).mean()):.2e}")
            ax.set_title(f"<{names[i]}(τ) {names[j]}(0)>", fontsize=8, color=INK)
            style(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau/\beta$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("CTHYB − ED", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


ed, run = collect()
fig_pairs = plot_pairs(ed, run)
fig_conserved = plot_conserved(ed, run)
if SAVE_AS:
    root, ext = os.path.splitext(SAVE_AS)
    fig_pairs.savefig(SAVE_AS)
    fig_conserved.savefig(f"{root}_conserved{ext}")
plt.show()
