# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# The vb_dimer benchmark, one figure per filling. Rows: G(tau), Re/Im Sigma, chi^zz(tau) and the
# chi residual against the reference; columns: beta. The reference is the exact ED at
# POINT = "ed" (J_intra = -J_inter) and CTHYB lang_firsov=True at POINT = "dca" (J_intra = 0),
# where no ED exists (-J indefinite). Knobs below; missing files are skipped.
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gfs import Gf, BlockGf  # noqa: F401
from triqs.stat.histograms import Histogram  # noqa: F401
# Unused, but they register the h5 readers for the solver objects in the run files.

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/vb_dimer"
POINT = "ed"           # "ed" (J_intra=-J_inter, exact ED) or "dca" (J_intra=0, no ED)
ROTATION = "site"      # "site" (the benchmark) or "none" (interaction local in the patch basis)
BETAS = [10.0, 100.0]
FILLINGS = [0.5, 0.75]  # one figure each: 0.5 = half filling, 0.75 = the doped runs (see collect)
T, TP, U, OMEGA_0 = 0.25, 0.0, 2.0, 1.0
N_PH = 3
ORBS = (0, 1)          # patch orbitals shown (spin up): K = 0 bonding, K = 1 antibonding
W_MAX = 15.0
SAVE_AS = None         # e.g. "vb_dimer_{point}_n{filling:g}.pdf"
# -----------------------------------------------------------------------------------------

if POINT == "ed":
    J_INTRA, J_INTER, BATH, V = -0.5, 0.5, "discrete", 0.5
else:
    J_INTRA, J_INTER, BATH, V = 0.0, 0.5, "dca", 0.5

# label -> (file prefix, extra name pieces)
SERIES = {"ED (exact)": ("ed_", (f"nph-{N_PH}",)),
          "CTHYB lf=True": ("cthyb_", ("lf-True",)),
          "CTHYB lf=False": ("cthyb_", ("lf-False",))}
if POINT != "ed":
    del SERIES["ED (exact)"]
STYLE = {"ED (exact)": dict(color="k", linestyle="-", linewidth=2.5),
         "CTHYB lf=True": dict(color="#e8710a", linestyle="--", linewidth=1.8),
         "CTHYB lf=False": dict(color="#c1121f", linestyle="-.", linewidth=1.5)}
PATCH_ALPHA = {0: 1.0, 1: 0.5}   # G(tau): K = 1 drawn fainter in the same style
PATCH_MARKER = {0: "o", 1: "s"}  # Sigma(iw): K = 0 filled circles, K = 1 open squares


def matsubara_style(st, K):
    """Marked points on a thin dotted line."""
    color = st.get("color", "k")
    return dict(color=color, linestyle=":", linewidth=0.8, marker=PATCH_MARKER[K], markersize=3.5,
                markerfacecolor=color if K == 0 else "none", markeredgewidth=0.9)


def candidates(prefix, beta, *extra):
    """Every result file for this prefix and beta, matching every tag piece that tells the
    J_ED and J_DCA files apart (they share a directory)."""
    if not os.path.isdir(DATA_DIR):
        return []
    bath = f"bath-V-{V}" if BATH == "discrete" else "bath-dca"
    required = (f"beta-{beta}_", f"Jintra-{J_INTRA}_", f"Jinter-{J_INTER}_", bath,
                f"rot-{ROTATION}") + extra
    return [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR))
            if f.startswith(prefix) and "_seed-" not in f and all(s in f for s in required)]


def load(path):
    with HDFArchive(path, "r") as A:
        return {k: A[k] for k in A.keys()}


def same_grid(tau_a, tau_b):
    """Whether two saved tau arrays are the same grid, so the residual needs no resampling."""
    return len(tau_a) == len(tau_b) and np.allclose(tau_a, tau_b)


def collect(beta):
    """`{filling: (mu, {label: run})}` at this beta: files grouped by the mu in their name, each
    group put at the filling nearest its <n> (ED's if present), newest file per group and per
    filling."""
    groups = {}
    for label, (prefix, extra) in SERIES.items():
        for path in candidates(prefix, beta, *extra):
            mu = re.search(r"_mu-([^_]+)_", os.path.basename(path)).group(1)
            groups.setdefault(mu, {}).setdefault(label, []).append(path)

    out = {}
    for mu, files in groups.items():
        paths = {label: max(found, key=os.path.getmtime) for label, found in files.items()}
        runs = {label: load(path) for label, path in paths.items()}
        first = next(label for label in SERIES if label in runs)
        n_mean = float(np.mean(runs[first]["density"]))
        filling = min(FILLINGS, key=lambda f: abs(f - n_mean))
        if abs(filling - n_mean) > 0.1:
            print(f"[beta={beta:g} {POINT}] mu={mu}: <n> = {n_mean:.3f} is near none of {FILLINGS} -- skipped")
            continue
        newest = max(os.path.getmtime(path) for path in paths.values())
        if filling in out:
            keep, drop = (mu, out[filling][0]) if newest > out[filling][2] else (out[filling][0], mu)
            print(f"[beta={beta:g} {POINT} n={filling:g}] mu={drop} also lands here; keeping the newer mu={keep}")
            if keep != mu:
                continue
        out[filling] = (mu, runs, newest)
        for label, path in paths.items():
            print(f"[beta={beta:g} {POINT} n={filling:g}] {label}: {os.path.basename(path)}")
    return {filling: (mu, runs) for filling, (mu, runs, _) in out.items()}


def make_figure(filling, columns):
    """One 5 x n_beta grid for this filling; `columns` is [(beta, mu, runs)]."""
    fig, axes = plt.subplots(5, len(columns), figsize=(5.4 * len(columns), 15), squeeze=False)

    for col, (beta, mu, runs) in enumerate(columns):
        ref_name = "ED (exact)" if "ED (exact)" in runs else "CTHYB lf=True"
        ref = runs.get(ref_name)
        residuals_drawn, mismatch = 0, None

        for name, r in runs.items():
            st = STYLE.get(name, dict())

            # Both solvers write G[a, tau] on a shared `tau`; one curve per patch.
            for K in ORBS:
                label = f"{name} K={K}"
                if "G" in r:
                    axes[0][col].plot(np.asarray(r["tau"]) / beta, np.asarray(r["G"])[K],
                                      label=label, **{**st, "alpha": PATCH_ALPHA[K]})
                if "Sigma" in r and "w_n" in r:
                    w = np.asarray(r["w_n"])
                    sig = np.asarray(r["Sigma"])
                    sig = sig[K] if sig.ndim == 2 else sig
                    keep = w <= W_MAX
                    axes[1][col].plot(w[keep], sig[keep].real, label=label, **matsubara_style(st, K))
                    axes[2][col].plot(w[keep], sig[keep].imag, label=label, **matsubara_style(st, K))

            # `corr` is <S^z_tot(tau) S^z_tot(0)> in both; ED's chi^zz_ij drawn thin underneath.
            if "corr" in r:
                axes[3][col].plot(np.asarray(r["tau_corr"]) / beta, np.asarray(r["corr"]),
                                  label=f"{name} " + r"$\chi^{zz}_{\rm tot}$", **st)
            chi = r.get("chi_zz")
            if chi is not None:
                chi = np.asarray(chi)
                for (i, j), ls in (((0, 0), (0, (4, 2))), ((0, 1), ":")):
                    axes[3][col].plot(np.asarray(r["tau"]) / beta, chi[i, j],
                                      color=st.get("color", "k"), linestyle=ls, linewidth=1.0,
                                      alpha=0.7, label=f"{name} " + rf"$\chi^{{zz}}_{{{i}{j}}}$")

            if ref is not None and name != ref_name:
                a = r.get("corr")
                b = ref.get("corr")
                # Point by point on the shared grid; files on another grid are skipped.
                if a is not None and b is not None and same_grid(r["tau_corr"], ref["tau_corr"]):
                    axes[4][col].plot(np.asarray(r["tau_corr"]) / beta, np.asarray(a) - np.asarray(b),
                                      label=f"{name} - {ref_name}", **{**st, "linewidth": 1.0})
                    residuals_drawn += 1
                elif a is not None and b is not None:
                    mismatch = (f"{name} on {len(r['tau_corr'])} tau points, {ref_name} on "
                                f"{len(ref['tau_corr'])}:\nrerun both on the shared grid")
                    print(f"[beta={beta:g} {POINT} n={filling:g}] {mismatch.replace(chr(10), ' ')}")

            print(f"[beta={beta:g} {POINT} n={filling:g} mu={mu}] {name:16s} "
                  + (f"sign={r['average_sign']:.3f}  " if "average_sign" in r else "")
                  + (f"<n>={np.mean(r['density']):.4f}" if "density" in r else ""))

        if "ED (exact)" in runs:
            trunc = runs["ED (exact)"].get("ed_truncation", float("nan"))
            axes[4][col].text(0.02, 0.06, f"ED phonon truncation: {float(trunc):.1e}",
                              transform=axes[4][col].transAxes, fontsize=8)
            why_empty = mismatch or "no CTHYB run at this point"
        elif POINT == "dca":
            axes[4][col].text(0.02, 0.8,
                              "No ED at this coupling:\n-J is not positive semidefinite,\n"
                              "so no real-boson Hamiltonian exists.\n"
                              "Reference is the lf True/False pair.",
                              transform=axes[4][col].transAxes, fontsize=8, va="center")
            why_empty = mismatch or f"only {', '.join(runs)} on disk:\nthe lf pair is incomplete"
        else:
            # An ED exists here; its file is missing.
            why_empty = "ED reference not found at this mu.\nRun:  sbatch run_vb_dimer.sh ed"
        if not residuals_drawn:
            axes[4][col].text(0.5, 0.45, why_empty, ha="center", va="center", fontsize=9,
                              color="#c1121f", transform=axes[4][col].transAxes)

        n_l = sorted({int(r["n_l"]) for r in runs.values() if "n_l" in r})
        axes[0][col].set_title(r"$\beta$" + f"={beta:g}, U={U:g}, "
                               + r"$J_{\rm intra}$" + f"={J_INTRA:g}, "
                               + r"$J_{\rm inter}$" + f"={J_INTER:g}, bath={BATH}\n"
                               + r"$\mu$" + f"={mu}, n={filling:g}, rotation={ROTATION}"
                               + (f", n_l={'/'.join(map(str, n_l))}" if n_l else ""))
        axes[4][col].axhline(0.0, color="k", linewidth=0.8)
        for row in (1, 2):
            axes[row][col].axhline(0.0, color="k", linewidth=0.5, alpha=0.3)
            axes[row][col].set_xlabel(r"$\omega_n$")
        for row in (3, 4):
            axes[row][col].set_xlabel(r"$\tau/\beta$")

    axes[0][0].set_ylabel(r"$G_K(\tau)$")
    axes[1][0].set_ylabel(r"$\mathrm{Re}\,\Sigma(i\omega_n)$")
    axes[2][0].set_ylabel(r"$\mathrm{Im}\,\Sigma(i\omega_n)$")
    axes[3][0].set_ylabel(r"$\chi^{zz}(\tau)$")
    axes[4][0].set_ylabel(r"$\Delta\chi^{zz}_{\rm tot}$ vs reference")
    for row in range(5):
        if axes[row][0].get_legend_handles_labels()[0]:
            axes[row][0].legend(fontsize=7)

    title = ("Valence-bond dimer, retarded real-space " + r"$\mathbf{S}\cdot\mathbf{S}$"
             + f", n={filling:g}" + "\n"
             + ("ED is exact at this coupling, so deviations are CTHYB errors"
                if POINT == "ed" else
                "No ED exists at this coupling (-J indefinite); reference is lf True vs False"))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


by_beta = {beta: collect(beta) for beta in BETAS}
drawn = 0
for filling in FILLINGS:
    columns = [(beta, *by_beta[beta][filling]) for beta in BETAS if filling in by_beta[beta]]
    if not columns:
        print(f"[{POINT} n={filling:g}] no files under {DATA_DIR} -- skipped")
        continue
    fig = make_figure(filling, columns)
    drawn += 1
    if SAVE_AS:
        name = SAVE_AS.format(point=POINT, filling=filling, rotation=ROTATION)
        fig.savefig(name, dpi=150, bbox_inches="tight")
        print(f"wrote {name}")
if not drawn:
    raise SystemExit(f"No files under {DATA_DIR} for POINT={POINT!r}, ROTATION={ROTATION!r}.\n"
                     "ED files come from:  sbatch run_vb_dimer.sh ed\n"
                     "(there is no ED for POINT='dca' -- see the header)")
plt.show()
