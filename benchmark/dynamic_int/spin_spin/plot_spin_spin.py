# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB and CTSEG against a reference (CTINT, else CTSEG) for the spin-spin benchmark, one
# figure per (beta, filling) in GRID. Rows: G(tau), Re and Im Sigma(iw_n), <Sz(tau)Sz(0)> and
# its residual against the reference; columns: the coupling cases on disk. Missing files are
# skipped. A series failing check_quality is still drawn, but cannot be the reference or set
# the y-limits.
#
#   python plot_spin_spin.py      (knobs below)
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin"
DATA_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin"  # on the cluster
# (beta, filling) -> one figure each. 0.5 is half filling, 0.75 the doped runs.
GRID = [(10.0, 0.5), (10.0, 0.75), (100.0, 0.5), (100.0, 0.75)]
U, J = 4.0, 1.0
CASES = [((1, 1), r"$\mathbf{S}\cdot\mathbf{S}$"),
         ((1, 0), r"$J_\perp$ only"),
         ((0, 1), r"$S_zS_z$ only")]
W_MAX = 15.0           # Matsubara axis limit
MIN_SIGN = 0.01        # below this a series is noise, see check_quality
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
MARKER = {"CTINT": "o", "CTSEG": "s", "CTHYB": "^", "CTHYB lf=False": "v"}


def matsubara_style(label, alt=False):
    """Marked points joined by a thin dotted line; the second estimator gets open markers."""
    color = STYLE[label]["color"]
    return dict(color=color, linestyle=":", linewidth=0.8, marker=MARKER[label], markersize=3.5,
                markerfacecolor="none" if alt else color, markeredgewidth=0.9,
                alpha=0.75 if alt else 1.0)


def load_case(jperp, szsz, beta, filling):
    runs = {}
    for label, (solver, tag) in SERIES.items():
        path = os.path.join(DATA_DIR, f"spin_spin_{solver}_b-{beta:g}_n-{filling:g}_J-{J:g}_jperp-{jperp:g}_szsz-{szsz:g}"
                                      + (f"_{tag}" if tag else "") + ".h5")
        if os.path.exists(path):
            with HDFArchive(path, "r") as A:
                runs[label] = {k: A[k] for k in A.keys()}
    return runs


def same_grid(tau_a, tau_b):
    """Whether two saved tau arrays are the same grid."""
    return len(tau_a) == len(tau_b) and np.allclose(tau_a, tau_b)


def check_quality(run):
    """`(ok, note)`: fails if the average sign is below MIN_SIGN (CTINT's collapses away from
    beta = 10, half filling) or if -G(0) - G(beta), exactly 1 for any model, is off by > 0.05."""
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
    """The 5 x n_cases figure for this (beta, filling), or None if no file exists."""
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

        # y-limits from the trusted series only, so an off-scale curve leaves the panel
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
                axes[1][col].plot(w[keep], sigma[keep].real, label=legend, **matsubara_style(label))
                axes[2][col].plot(w[keep], sigma[keep].imag, label=legend, **matsubara_style(label))
                if ok:
                    note(1, sigma[keep].real)
                    note(2, sigma[keep].imag)
                # Second Sigma estimator
                if run.get("Sigma_alt") is not None:
                    alt = np.asarray(run["Sigma_alt"])
                    axes[1][col].plot(w[keep], alt[keep].real, **matsubara_style(label, alt=True))
                    axes[2][col].plot(w[keep], alt[keep].imag, **matsubara_style(label, alt=True))
                    if ok:
                        note(1, alt[keep].real)
                        note(2, alt[keep].imag)

            if "corr" in run:
                axes[3][col].plot(run["tau_corr"], run["corr"], label=legend, **style)
                if ok:
                    note(3, run["corr"])
                if run.get("corr_alt") is not None:
                    # Saved without its tau: the uniform bosonic mesh on [0, beta]
                    alt = np.asarray(run["corr_alt"])
                    axes[3][col].plot(np.linspace(0.0, run["beta"], len(alt)), alt,
                                      color=style["color"], **ALT_STYLE)
                    if ok:
                        note(3, alt)
                if ref is not None and label != ref_label and ok:
                    # Point by point on the shared grid of common/grids.py; other grids are skipped
                    if same_grid(run["tau_corr"], ref["tau_corr"]):
                        axes[4][col].plot(run["tau_corr"], run["corr"] - ref["corr"], label=label,
                                          **{**style, "linewidth": 1.0})
                    else:
                        print(f"[beta={beta:g} n={filling:g} jperp={jperp} szsz={szsz}] {label}: "
                              f"{len(run['tau_corr'])} tau points against {ref_label}'s "
                              f"{len(ref['tau_corr'])} -- residual skipped, rerun on the shared grid")

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

        # At half filling Re Sigma = mu exactly
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
        # Skip empty legends, which matplotlib warns about
        if axes[row][0].get_legend_handles_labels()[0]:
            axes[row][0].legend(fontsize=8)

    fig.suptitle(r"Single-orbital retarded spin-spin, $\beta$" + f"={beta:g}, n={filling:g}"
                 "\nsecond estimator of the same quantity (bounds the systematic): dotted in tau, "
                 "open markers in iw_n; "
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
