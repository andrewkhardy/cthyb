# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB and CTSEG against CTINT for the single-orbital retarded spin-spin benchmark.
# Rows: G(tau), Re Sigma(iw_n), Im Sigma(iw_n), <Sz(tau)Sz(0)>, and that correlator's
# residual against CTINT. Columns: the coupling cases present on disk.
#
# Knobs are hardcoded below so this pastes straight into a notebook; missing files are
# skipped, so a partially finished grid still plots.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import io

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin"
BETA = 10.0
FILLING = 0.5          # 0.5 = half filling, 0.75 = the doped runs
U, J = 4.0, 1.0
CASES = [((1, 1), r"$\mathbf{S}\cdot\mathbf{S}$"),
         ((1, 0), r"$J_\perp$ only"),
         ((0, 1), r"$S_zS_z$ only")]
W_MAX = 15.0           # Matsubara axis limit; Sigma is plotted raw, never tail-fitted
RESIDUAL_POINTS = 61   # coarse common grid for the residual panel, see below
SAVE_AS = None         # e.g. "spin_spin_b10_n05.pdf"
# -----------------------------------------------------------------------------------------

# series label -> (solver, filename tag). CTINT is the reference.
SERIES = {
    "CTINT": ("ctint", ""),
    "CTSEG": ("ctseg", ""),
    "CTHYB": ("cthyb", "lf-True"),
    "CTHYB lf=False": ("cthyb", "lf-False"),
}
STYLE = {
    "CTINT": dict(color="#2a9d3f", linestyle="-", linewidth=3.0),
    "CTSEG": dict(color="#1f6feb", linestyle="--", linewidth=2.0),
    "CTHYB": dict(color="#e8710a", linestyle=":", linewidth=2.2),
    "CTHYB lf=False": dict(color="#c1121f", linestyle="-.", linewidth=1.5),
}
ALT_STYLE = dict(linewidth=1.0, alpha=0.75, linestyle=(0, (1, 1)))


def load_case(jperp, szsz):
    runs = {}
    for label, (solver, tag) in SERIES.items():
        full_tag = f"J-{J:g}_jperp-{jperp:g}_szsz-{szsz:g}" + (f"_{tag}" if tag else "")
        path = io.output_file(DATA_DIR, "spin_spin", solver, BETA, FILLING, tag=full_tag)
        run = io.load(path)
        if run is not None:
            runs[label] = run
    return runs


cases = [(c, t) for c, t in CASES if load_case(*c)]
if not cases:
    raise SystemExit(f"No files found under {DATA_DIR} for beta={BETA:g}, filling={FILLING:g}.\n"
                     "Check DATA_DIR, BETA and FILLING at the top of this script.")

fig, axes = plt.subplots(5, len(cases), figsize=(5.2 * len(cases), 15), squeeze=False)

for col, ((jperp, szsz), title) in enumerate(cases):
    runs = load_case(jperp, szsz)
    ref = runs.get("CTINT")

    for label, run in runs.items():
        style = STYLE[label]
        axes[0][col].plot(run["tau_G"], run["G"], label=label, **style)

        if "Sigma" in run:
            w, sigma = run["w_n"], run["Sigma"]
            keep = w <= W_MAX
            axes[1][col].plot(w[keep], sigma[keep].real, label=label, **style)
            axes[2][col].plot(w[keep], sigma[keep].imag, label=label, **style)
            # Second, independent Sigma route -- the pair bounds the systematic.
            if run.get("Sigma_alt") is not None:
                alt = np.asarray(run["Sigma_alt"])
                axes[1][col].plot(w[keep], alt[keep].real, color=style["color"], **ALT_STYLE)
                axes[2][col].plot(w[keep], alt[keep].imag, color=style["color"], **ALT_STYLE)

        if "corr" in run:
            axes[3][col].plot(run["tau_corr"], run["corr"], label=label, **style)
            if run.get("corr_alt") is not None:
                axes[3][col].plot(run["raw"]["tau_corr_alt"] if "raw" in run else run["tau_corr"],
                                  run["corr_alt"], color=style["color"], **ALT_STYLE)
            if ref is not None and label != "CTINT":
                # On a coarse common grid: the raw tau grids differ between solvers (2001
                # vs 501 points here), so a point-by-point residual is dominated by
                # interpolation noise and is unreadable. Binning to RESIDUAL_POINTS shows
                # the systematic offset, which is what this panel is for.
                grid = np.linspace(0.0, BETA, RESIDUAL_POINTS)
                residual = (np.interp(grid, run["tau_corr"], run["corr"])
                            - np.interp(grid, ref["tau_corr"], ref["corr"]))
                axes[4][col].plot(grid, residual, label=label, marker="o", markersize=2.5, **style)

        order = io.mean_order(run.get("pert_order_dyn"))
        n_mean = np.mean(run["density"]) if "density" in run else float("nan")
        print(f"[beta={BETA:g} n={FILLING:g} jperp={jperp} szsz={szsz}] {label:15s} "
              f"sign={run.get('average_sign', float('nan')):.3f}  <k_dyn>={order:6.3f}  "
              f"<n>={n_mean:.4f}  <SzSz>(beta/2)="
              f"{np.interp(BETA / 2, run['tau_corr'], run['corr']):.5f}")

    axes[0][col].set_title(f"{title}\n" + r"$\beta$" + f"={BETA:g}, U={U:g}, J={J:g}, n={FILLING:g}")
    axes[4][col].axhline(0.0, color="k", linewidth=0.8)
    for row in (1, 2, 4):
        axes[row][col].axhline(0.0, color="k", linewidth=0.5, alpha=0.3)
    axes[4][col].set_xlabel(r"$\tau$")
    axes[3][col].set_xlabel(r"$\tau$")
    for row in (1, 2):
        axes[row][col].set_xlabel(r"$\omega_n$")

# Exact identity at half filling: Re Sigma = mu at every frequency (see
# common/selfenergy.diagnose). Draw it as the target the curves should sit on.
if abs(FILLING - 0.5) < 1e-12:
    for col in range(len(cases)):
        runs = load_case(*cases[col][0])
        if runs:
            mu = float(np.atleast_1d(list(runs.values())[0]["mu"])[0])
            axes[1][col].axhline(mu, color="k", linewidth=0.9, linestyle=(0, (4, 3)),
                                 label=r"exact: $\mathrm{Re}\,\Sigma=\mu$")

axes[0][0].set_ylabel(r"$G_\uparrow(\tau)$")
axes[1][0].set_ylabel(r"$\mathrm{Re}\,\Sigma(i\omega_n)$")
axes[2][0].set_ylabel(r"$\mathrm{Im}\,\Sigma(i\omega_n)$")
axes[3][0].set_ylabel(r"$\langle S_z(\tau)S_z(0)\rangle$")
axes[4][0].set_ylabel(r"$\Delta\langle S_zS_z\rangle$ vs CTINT")
for row in range(5):
    axes[row][0].legend(fontsize=8)

fig.suptitle("Single-orbital retarded spin-spin: CTHYB and CTSEG vs CTINT\n"
             "dotted = second estimator of the same quantity (bounds the systematic)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=150, bbox_inches="tight")
    print(f"wrote {SAVE_AS}")
plt.show()
