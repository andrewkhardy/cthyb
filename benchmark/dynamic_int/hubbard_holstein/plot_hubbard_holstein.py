# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB and CTSEG against CTINT for the single-orbital Hubbard-Holstein benchmark.
# Rows: G(tau), Re Sigma, Im Sigma, <N(tau)N(0)>, and that correlator's residual vs CTINT.
# Columns: the temperatures present on disk, so the beta dependence is side by side.
#
# Knobs hardcoded below so this pastes into a notebook; missing files are skipped.
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/hubbard_holstein"
BETAS = [10.0, 100.0]
FILLING = 0.5          # 0.5 = half filling, 0.75 = the doped runs
U, G, OMEGA_0 = 4.0, 0.7, 1.0   # set G=0.3 for the weak-coupling point where CTINT is also available
W_MAX = 15.0           # Sigma is plotted raw on the Matsubara points, never tail-fitted
RESIDUAL_POINTS = 61
SAVE_AS = None
# -----------------------------------------------------------------------------------------

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


def load_beta(beta):
    runs = {}
    for label, (solver, tag) in SERIES.items():
        path = os.path.join(DATA_DIR, f"hubbard_holstein_{solver}_b-{beta:g}_n-{FILLING:g}_g-{G:g}_w0-{OMEGA_0:g}"
                                      + (f"_{tag}" if tag else "") + ".h5")
        if os.path.exists(path):
            with HDFArchive(path, "r") as A:
                runs[label] = {k: A[k] for k in A.keys()}
    return runs


betas = [b for b in BETAS if load_beta(b)]
if not betas:
    raise SystemExit(f"No files under {DATA_DIR} for filling={FILLING:g}, g={G:g}, omega_0={OMEGA_0:g}.\n"
                     "Check DATA_DIR and the knobs at the top of this script.")

fig, axes = plt.subplots(5, len(betas), figsize=(5.2 * len(betas), 15), squeeze=False)

for col, beta in enumerate(betas):
    runs = load_beta(beta)
    ref = runs.get("CTINT")

    for label, run in runs.items():
        style = STYLE[label]
        # tau/beta on the x axis, so the two temperatures are directly comparable.
        axes[0][col].plot(run["tau_G"] / beta, run["G"], label=label, **style)

        if "Sigma" in run:
            w, sigma = run["w_n"], run["Sigma"]
            keep = w <= W_MAX
            axes[1][col].plot(w[keep], sigma[keep].real, label=label, **style)
            axes[2][col].plot(w[keep], sigma[keep].imag, label=label, **style)
            if run.get("Sigma_alt") is not None:
                alt = np.asarray(run["Sigma_alt"])
                axes[1][col].plot(w[keep], alt[keep].real, color=style["color"], **ALT_STYLE)
                axes[2][col].plot(w[keep], alt[keep].imag, color=style["color"], **ALT_STYLE)

        if "corr" in run:
            axes[3][col].plot(run["tau_corr"] / beta, run["corr"], label=label, **style)
            if run.get("corr_alt") is not None:
                # corr_alt is on the BOSONIC tau mesh (Q_tau, n_tau_bosonic points), corr on
                # O_tau's fermionic one, so it needs its own x. The bosonic mesh is uniform on
                # [0, beta] and linspace reproduces it to 2e-15.
                alt = np.asarray(run["corr_alt"])
                axes[3][col].plot(np.linspace(0.0, beta, len(alt)) / beta, alt,
                                  color=style["color"], **ALT_STYLE)
            if ref is not None and label != "CTINT":
                grid = np.linspace(0.0, beta, RESIDUAL_POINTS)
                residual = (np.interp(grid, run["tau_corr"], run["corr"])
                            - np.interp(grid, ref["tau_corr"], ref["corr"]))
                axes[4][col].plot(grid / beta, residual, label=label,
                                  marker="o", markersize=2.5, **style)

        n_mean = np.mean(run["density"]) if "density" in run else float("nan")
        h = np.asarray(run.get("pert_order_dyn", [np.nan]), dtype=float)
        print(f"[beta={beta:g} n={FILLING:g}] {label:15s} sign={run.get('average_sign', float('nan')):.3f}  "
              f"<k_dyn>={(np.arange(len(h)) * h).sum() / h.sum():6.3f}  <n>={n_mean:.4f}  "
              f"<NN>(beta/2)={np.interp(beta / 2, run['tau_corr'], run['corr']):.5f}")

    # Exact at half filling: Re Sigma = mu at every frequency (common/selfenergy.diagnose).
    if abs(FILLING - 0.5) < 1e-12 and runs:
        mu = float(np.atleast_1d(list(runs.values())[0]["mu"])[0])
        axes[1][col].axhline(mu, color="k", linewidth=0.9, linestyle=(0, (4, 3)),
                             label=r"exact: $\mathrm{Re}\,\Sigma=\mu$")

    axes[0][col].set_title(r"$\beta$" + f"={beta:g}, U={U:g}, g={G:g}, "
                           + r"$\omega_0$" + f"={OMEGA_0:g}, n={FILLING:g}\n"
                           + r"polaron shift $g^2/\omega_0^2$=" + f"{G ** 2 / OMEGA_0 ** 2:.2f}")
    axes[4][col].axhline(0.0, color="k", linewidth=0.8)
    for row in (1, 2):
        axes[row][col].axhline(0.0, color="k", linewidth=0.5, alpha=0.3)
        axes[row][col].set_xlabel(r"$\omega_n$")
    for row in (3, 4):
        axes[row][col].set_xlabel(r"$\tau/\beta$")

axes[0][0].set_ylabel(r"$G_\uparrow(\tau)$")
axes[1][0].set_ylabel(r"$\mathrm{Re}\,\Sigma(i\omega_n)$")
axes[2][0].set_ylabel(r"$\mathrm{Im}\,\Sigma(i\omega_n)$")
axes[3][0].set_ylabel(r"$\langle N(\tau)N(0)\rangle$")
axes[4][0].set_ylabel(r"$\Delta\langle NN\rangle$ vs CTINT")
for row in range(5):
    axes[row][0].legend(fontsize=8)

fig.suptitle("Single-orbital Hubbard-Holstein: CTHYB and CTSEG vs CTINT\n"
             "dotted = second estimator of the same quantity; "
             "CTHYB lf=False is the analytic-vs-stochastic cross-check", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=150, bbox_inches="tight")
    print(f"wrote {SAVE_AS}")
plt.show()
