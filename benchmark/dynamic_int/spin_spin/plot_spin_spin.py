# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB and CTSEG against a reference for the single-orbital retarded spin-spin benchmark.
# Rows: G(tau), Re Sigma(iw_n), Im Sigma(iw_n), <Sz(tau)Sz(0)>, and that correlator's
# residual against the reference. Columns: the coupling cases present on disk.
#
# One figure per (beta, filling) in GRID -- four of them as configured, which is the full
# set the submit script produces. A combination with no files on disk is skipped with a
# message rather than raising, so the other three still draw.
#
# Every series on disk is always drawn. CTINT's sign collapses away from (beta=10, n=0.5)
# and what it returns there is not a Green function -- see check_quality -- so the y-limits
# are set from the trustworthy series only and a blown-up curve is allowed to leave the
# panel rather than flatten everything else in it. Such a series is also barred from being
# the reference; REFERENCE_ORDER then falls through to CTSEG, which is sign-free here.
#
# Knobs are hardcoded below so this pastes straight into a notebook; missing files are
# skipped, so a partially finished grid still plots.
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin"
# (beta, filling) -> one figure each. 0.5 is half filling, 0.75 the doped runs.
GRID = [(10.0, 0.5), (10.0, 0.75), (100.0, 0.5), (100.0, 0.75)]
U, J = 4.0, 1.0
CASES = [((1, 1), r"$\mathbf{S}\cdot\mathbf{S}$"),
         ((1, 0), r"$J_\perp$ only"),
         ((0, 1), r"$S_zS_z$ only")]
W_MAX = 15.0           # Matsubara axis limit; Sigma is plotted raw, never tail-fitted
RESIDUAL_POINTS = 61   # coarse common grid for the residual panel, see below
MIN_SIGN = 0.01        # below this a series is noise, not data -- see check_quality
REFERENCE_ORDER = ["CTINT", "CTSEG"]   # first one that passes check_quality is the reference
SAVE_AS = None         # e.g. "spin_spin_b{beta:g}_n{filling:g}.pdf"
# -----------------------------------------------------------------------------------------

# series label -> (solver, filename tag).
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


def load_case(jperp, szsz, beta, filling):
    runs = {}
    for label, (solver, tag) in SERIES.items():
        path = os.path.join(DATA_DIR, f"spin_spin_{solver}_b-{beta:g}_n-{filling:g}_J-{J:g}_jperp-{jperp:g}_szsz-{szsz:g}"
                                      + (f"_{tag}" if tag else "") + ".h5")
        if os.path.exists(path):
            with HDFArchive(path, "r") as A:
                runs[label] = {k: A[k] for k in A.keys()}
    return runs


def check_quality(run):
    """`(ok, note)` -- is this series data, or is it noise wearing a solver's name?

    Two independent tests, both cheap and both decisive:

      * the average sign. CTINT's collapses both at beta = 100 and away from half filling
        -- on these files 2.5e-4 at (beta=10, n=0.75) and 9.4e-5 at (beta=100, n=0.5),
        against 0.365 at (beta=10, n=0.5). CT-INT measures <O s>/<s>, so a denominator
        going to zero blows the variance of the ratio up without bound.

      * the anticommutator, -G(0) - G(beta) = 1 exactly, for any Hamiltonian and any bath.
        The two collapsed runs give 1.42 and 1.63, with max |G(tau)| of 1.23 and 14.14
        where a fermionic G cannot exceed 1, and densities up to 16.2 per spin-orbital.

    A merely noisy series passes both (sign 0.365, jump 1.0008) and is treated as data.
    Failing either bars a series from being the reference and from setting the y-limits,
    but NOT from being drawn -- see the module docstring.
    """
    if run is None:
        return False, "absent"
    sign = float(run.get("average_sign", 0.0))
    if sign < MIN_SIGN:
        return False, f"sign {sign:.0e}"
    jump = float(-np.asarray(run["G"])[0] - np.asarray(run["G"])[-1])
    if abs(jump - 1.0) > 0.05:
        return False, f"$-G(0)-G(\\beta)$ = {jump:.2f}"
    return True, ""


def make_figure(beta, filling):
    """One 5 x n_cases grid for this (beta, filling). Returns the figure, or None if the
    combination has no files at all -- the caller reports it and moves on."""
    cases = [(c, t) for c, t in CASES if load_case(*c, beta, filling)]
    if not cases:
        return None

    fig, axes = plt.subplots(5, len(cases), figsize=(5.2 * len(cases), 15), squeeze=False)

    for col, ((jperp, szsz), title) in enumerate(cases):
        runs = load_case(jperp, szsz, beta, filling)
        quality = {lab: check_quality(run) for lab, run in runs.items()}
        trusted = [lab for lab, (ok, _) in quality.items() if ok]
        ref_label = next((lab for lab in REFERENCE_ORDER if lab in trusted), None)
        ref = runs.get(ref_label)

        # y-limits come from the trusted series only, so an off-scale curve is visible as
        # "it leaves the panel" instead of compressing every other curve into a flat line.
        span = {row: [np.inf, -np.inf] for row in range(4)}

        def note(row, values):
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                span[row][0] = min(span[row][0], v.min())
                span[row][1] = max(span[row][1], v.max())

        for label, run in runs.items():
            style = STYLE[label]
            ok, why = quality[label]
            legend = label if ok else f"{label}  [{why}]"
            axes[0][col].plot(run["tau_G"], run["G"], label=legend, **style)
            if ok:
                note(0, run["G"])

            if "Sigma" in run:
                w, sigma = run["w_n"], run["Sigma"]
                keep = w <= W_MAX
                axes[1][col].plot(w[keep], sigma[keep].real, label=legend, **style)
                axes[2][col].plot(w[keep], sigma[keep].imag, label=legend, **style)
                if ok:
                    note(1, sigma[keep].real)
                    note(2, sigma[keep].imag)
                # Second, independent Sigma route -- the pair bounds the systematic.
                if run.get("Sigma_alt") is not None:
                    alt = np.asarray(run["Sigma_alt"])
                    axes[1][col].plot(w[keep], alt[keep].real, color=style["color"], **ALT_STYLE)
                    axes[2][col].plot(w[keep], alt[keep].imag, color=style["color"], **ALT_STYLE)
                    if ok:
                        note(1, alt[keep].real)
                        note(2, alt[keep].imag)

            if "corr" in run:
                axes[3][col].plot(run["tau_corr"], run["corr"], label=legend, **style)
                if ok:
                    note(3, run["corr"])
                if run.get("corr_alt") is not None:
                    # corr_alt lives on the BOSONIC tau mesh (Q_tau, n_tau_bosonic points) while
                    # corr lives on O_tau's fermionic one -- 2001 vs 4096 for these files -- so it
                    # needs its own x. The mesh is uniform on [0, beta], and linspace reproduces
                    # it to 2e-15.
                    alt = np.asarray(run["corr_alt"])
                    axes[3][col].plot(np.linspace(0.0, run["beta"], len(alt)), alt,
                                      color=style["color"], **ALT_STYLE)
                    if ok:
                        note(3, alt)
                if ref is not None and label != ref_label and ok:
                    # On a coarse common grid: the raw tau grids differ between solvers (2001
                    # vs 501 points here), so a point-by-point residual is dominated by
                    # interpolation noise and is unreadable. Binning to RESIDUAL_POINTS shows
                    # the systematic offset, which is what this panel is for.
                    grid = np.linspace(0.0, beta, RESIDUAL_POINTS)
                    residual = (np.interp(grid, run["tau_corr"], run["corr"])
                                - np.interp(grid, ref["tau_corr"], ref["corr"]))
                    axes[4][col].plot(grid, residual, label=label, marker="o", markersize=2.5, **style)

            h = np.asarray(run.get("pert_order_dyn", [np.nan]), dtype=float)
            order = (np.arange(len(h)) * h).sum() / h.sum()
            n_mean = np.mean(run["density"]) if "density" in run else float("nan")
            flag = "" if ok else f"   <-- NOT TRUSTED ({why})"
            print(f"[beta={beta:g} n={filling:g} jperp={jperp} szsz={szsz}] {label:15s} "
                  f"sign={run.get('average_sign', float('nan')):.3f}  <k_dyn>={order:6.3f}  "
                  f"<n>={n_mean:.4f}  <SzSz>(beta/2)="
                  f"{np.interp(beta / 2, run['tau_corr'], run['corr']):.5f}{flag}")

        for row in range(4):
            lo, hi = span[row]
            if np.isfinite(lo) and np.isfinite(hi):
                pad = 0.08 * max(hi - lo, 1e-12)
                axes[row][col].set_ylim(lo - pad, hi + pad)

        if ref_label is None:
            axes[4][col].text(0.5, 0.5, "no usable reference for this point",
                              ha="center", va="center", fontsize=9, color="#c1121f",
                              transform=axes[4][col].transAxes)
        else:
            # As a title, not in-axes text: the panel's zero line sits where text would go.
            axes[4][col].set_title(f"reference: {ref_label}", fontsize=8, loc="left")

        axes[0][col].set_title(f"{title}\n" + r"$\beta$"
                               + f"={beta:g}, U={U:g}, J={J:g}, n={filling:g}")
        axes[4][col].axhline(0.0, color="k", linewidth=0.8)
        for row in (1, 2, 4):
            axes[row][col].axhline(0.0, color="k", linewidth=0.5, alpha=0.3)
        axes[4][col].set_xlabel(r"$\tau$")
        axes[3][col].set_xlabel(r"$\tau$")
        for row in (1, 2):
            axes[row][col].set_xlabel(r"$\omega_n$")

        # Exact identity at half filling: Re Sigma = mu at every frequency (see
        # common/selfenergy.diagnose). Draw it as the target the curves should sit on.
        if abs(filling - 0.5) < 1e-12 and trusted:
            mu = float(np.atleast_1d(runs[trusted[0]]["mu"])[0])
            axes[1][col].axhline(mu, color="k", linewidth=0.9, linestyle=(0, (4, 3)),
                                 label=r"exact: $\mathrm{Re}\,\Sigma=\mu$")

    axes[0][0].set_ylabel(r"$G_\uparrow(\tau)$")
    axes[1][0].set_ylabel(r"$\mathrm{Re}\,\Sigma(i\omega_n)$")
    axes[2][0].set_ylabel(r"$\mathrm{Im}\,\Sigma(i\omega_n)$")
    axes[3][0].set_ylabel(r"$\langle S_z(\tau)S_z(0)\rangle$")
    axes[4][0].set_ylabel(r"$\Delta\langle S_zS_z\rangle$ vs reference")
    for row in range(5):
        # Only where something was actually drawn, else matplotlib warns about an empty legend.
        if axes[row][0].get_legend_handles_labels()[0]:
            axes[row][0].legend(fontsize=8)

    fig.suptitle(r"Single-orbital retarded spin-spin, $\beta$" + f"={beta:g}, n={filling:g}"
                 "\ndotted = second estimator of the same quantity (bounds the systematic); "
                 "[...] marks a series that failed the sign / sum-rule check",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


for beta, filling in GRID:
    fig = make_figure(beta, filling)
    if fig is None:
        print(f"[beta={beta:g} n={filling:g}] no files under {DATA_DIR} -- skipped")
        continue
    if SAVE_AS:
        name = SAVE_AS.format(beta=beta, filling=filling)
        fig.savefig(name, dpi=150, bbox_inches="tight")
        print(f"wrote {name}")
plt.show()
