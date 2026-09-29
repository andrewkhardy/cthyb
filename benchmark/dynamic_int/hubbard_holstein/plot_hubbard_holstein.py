# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB, CTSEG and CTINT for the single-orbital Hubbard-Holstein benchmark.
# Rows: G(tau), Re Sigma, Im Sigma, <N(tau)N(0)>, and that correlator's residual against the
# reference. Columns: the temperatures present on disk, so the beta dependence is side by side.
# One figure per filling in FILLINGS; a filling with no files on disk is skipped.
#
# The reference is the first of REFERENCE_ORDER that passes check_quality, so at G = 0.7,
# where there is no CTINT, CTHYB is measured against CTSEG instead of leaving the panel empty.
# A series that fails the check is still drawn, marked in the legend, but it is barred from
# being the reference and from setting the y-limits, so a blown-up curve leaves the panel
# rather than flattening everything else in it.
#
# Knobs hardcoded below so this pastes into a notebook; missing files are skipped.
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/hubbard_holstein"
DATA_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein"  # on the cluster

BETAS = [10.0, 100.0]
FILLINGS = [0.5, 0.75]  # one figure each: 0.5 = half filling, 0.75 = the doped runs
U, G, OMEGA_0 = 4.0, 0.7, 1.0   # set G=0.3 for the weak-coupling point where CTINT is also available
W_MAX = 15.0           # Sigma is plotted raw on the Matsubara points, never tail-fitted
MIN_SIGN = 0.01        # below this a series is noise, not data -- see check_quality
DENSITY_TOL = 0.01     # |<n> - filling| above this means a different model -- see check_quality
REFERENCE_ORDER = ["CTINT", "CTSEG"]   # first one that passes check_quality is the reference
SAVE_AS = None         # e.g. "hubbard_holstein_g{g:g}_n{filling:g}.pdf"
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
MARKER = {"CTINT": "o", "CTSEG": "s", "CTHYB": "^", "CTHYB lf=False": "v"}


def matsubara_style(label, alt=False):
    """Every Matsubara point marked, joined by a thin dotted line: the data are discrete, and
    a solid curve would hide where the points actually are. The second estimator gets the
    same marker, open."""
    color = STYLE[label]["color"]
    return dict(color=color, linestyle=":", linewidth=0.8, marker=MARKER[label], markersize=3.5,
                markerfacecolor="none" if alt else color, markeredgewidth=0.9,
                alpha=0.75 if alt else 1.0)


def load_beta(beta, filling):
    runs = {}
    for label, (solver, tag) in SERIES.items():
        path = os.path.join(DATA_DIR, f"hubbard_holstein_{solver}_b-{beta:g}_n-{filling:g}_g-{G:g}_w0-{OMEGA_0:g}"
                                      + (f"_{tag}" if tag else "") + ".h5")
        if os.path.exists(path):
            with HDFArchive(path, "r") as A:
                runs[label] = {k: A[k] for k in A.keys()}
    return runs


def same_grid(tau_a, tau_b):
    """Whether two saved tau arrays are the same grid, so the residual needs no resampling."""
    return len(tau_a) == len(tau_b) and np.allclose(tau_a, tau_b)


def check_quality(run, filling):
    """`(ok, note)` -- is this series data for this model, or something else?

      * the average sign. CT-INT measures <O s>/<s>, so a sign near zero blows the variance
        of every observable up without bound.
      * the anticommutator, -G(0) - G(beta) = 1 exactly, for any Hamiltonian and bath.
      * the density. Away from half filling mu is pinned per coupling, so a run whose <n>
        misses the filling in its filename solved a different model -- e.g. CTINT at
        G = 0.3 run with the mu calibrated for G = 0.7, which lands at n = 0.65.

    Failing any of them bars a series from being the reference and from setting the
    y-limits, but NOT from being drawn.
    """
    sign = float(np.real(run.get("average_sign", 0.0)))
    if sign < MIN_SIGN:
        return False, f"sign {sign:.0e}"
    jump = float(-np.asarray(run["G"])[0] - np.asarray(run["G"])[-1])
    if abs(jump - 1.0) > 0.05:
        return False, f"$-G(0)-G(\\beta)$ = {jump:.2f}"
    if "density" in run:
        n_mean = float(np.mean(run["density"]))
        if abs(n_mean - filling) > DENSITY_TOL:
            return False, f"<n> = {n_mean:.3f}"
    return True, ""


def make_figure(filling):
    """One 5 x n_beta grid for this filling, or None if no file exists for it."""
    betas = [b for b in BETAS if load_beta(b, filling)]
    if not betas:
        return None

    fig, axes = plt.subplots(5, len(betas), figsize=(5.2 * len(betas), 15), squeeze=False)

    for col, beta in enumerate(betas):
        runs = load_beta(beta, filling)
        quality = {label: check_quality(run, filling) for label, run in runs.items()}
        trusted = [label for label, (ok, _) in quality.items() if ok]
        ref_label = next((label for label in REFERENCE_ORDER if label in trusted), None)
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

        residuals_drawn = 0
        for label, run in runs.items():
            style = STYLE[label]
            ok, why = quality[label]
            legend = label if ok else f"{label}  [{why}]"
            # tau/beta on the x axis, so the two temperatures are directly comparable.
            axes[0][col].plot(run["tau_G"] / beta, run["G"], label=legend, **style)
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
                if run.get("Sigma_alt") is not None:
                    alt = np.asarray(run["Sigma_alt"])
                    axes[1][col].plot(w[keep], alt[keep].real, **matsubara_style(label, alt=True))
                    axes[2][col].plot(w[keep], alt[keep].imag, **matsubara_style(label, alt=True))
                    if ok:
                        note(1, alt[keep].real)
                        note(2, alt[keep].imag)

            if "corr" in run:
                axes[3][col].plot(run["tau_corr"] / beta, run["corr"], label=legend, **style)
                if ok:
                    note(3, run["corr"])
                if run.get("corr_alt") is not None:
                    # Saved without its own tau. The mesh is uniform on [0, beta], which
                    # linspace reproduces to 2e-15.
                    alt = np.asarray(run["corr_alt"])
                    axes[3][col].plot(np.linspace(0.0, beta, len(alt)) / beta, alt,
                                      color=style["color"], **ALT_STYLE)
                    if ok:
                        note(3, alt)
                if ref is not None and label != ref_label and ok:
                    # Point by point: every solver writes its correlator on the grid of
                    # common/grids.py. A file from before that is skipped, not resampled.
                    if same_grid(run["tau_corr"], ref["tau_corr"]):
                        axes[4][col].plot(run["tau_corr"] / beta, run["corr"] - ref["corr"],
                                          label=label, **{**style, "linewidth": 1.0})
                        residuals_drawn += 1
                    else:
                        print(f"[beta={beta:g} n={filling:g}] {label}: {len(run['tau_corr'])} tau "
                              f"points against {ref_label}'s {len(ref['tau_corr'])} -- residual "
                              "skipped, rerun on the shared grid")

            n_mean = np.mean(run["density"]) if "density" in run else float("nan")
            h = np.asarray(run.get("pert_order_dyn", [np.nan]), dtype=float)
            flag = "" if ok else f"   <-- NOT TRUSTED ({why})"
            print(f"[beta={beta:g} n={filling:g}] {label:15s} "
                  f"sign={float(np.real(run.get('average_sign', np.nan))):.3f}  "
                  f"<k_dyn>={(np.arange(len(h)) * h).sum() / h.sum():6.3f}  <n>={n_mean:.4f}  "
                  f"<NN>(beta/2)={np.interp(beta / 2, run['tau_corr'], run['corr']):.5f}{flag}")

        for row in range(4):
            lo, hi = span[row]
            if np.isfinite(lo) and np.isfinite(hi):
                pad = 0.08 * max(hi - lo, 1e-12)
                axes[row][col].set_ylim(lo - pad, hi + pad)

        # Say why the residual panel is empty rather than leave a blank box.
        if ref_label is None:
            why_empty = "no usable reference at this point\n(" + ", ".join(
                f"{label}: {why}" for label, (ok, why) in quality.items() if not ok) + ")"
        elif residuals_drawn == 0:
            why_empty = f"only {ref_label} is usable here,\nnothing to compare it with"
        else:
            why_empty = None
        if why_empty:
            axes[4][col].text(0.5, 0.5, why_empty, ha="center", va="center", fontsize=9,
                              color="#c1121f", transform=axes[4][col].transAxes)
        if ref_label is not None:
            # As a title, not in-axes text: the panel's zero line sits where text would go.
            axes[4][col].set_title(f"reference: {ref_label}", fontsize=8, loc="left")

        # Exact at half filling: Re Sigma = mu at every frequency (common/selfenergy.diagnose).
        if abs(filling - 0.5) < 1e-12 and trusted:
            mu = float(np.atleast_1d(runs[trusted[0]]["mu"])[0])
            axes[1][col].axhline(mu, color="k", linewidth=0.9, linestyle=(0, (4, 3)),
                                 label=r"exact: $\mathrm{Re}\,\Sigma=\mu$")

        axes[0][col].set_title(r"$\beta$" + f"={beta:g}, U={U:g}, g={G:g}, "
                               + r"$\omega_0$" + f"={OMEGA_0:g}, n={filling:g}\n"
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
    axes[4][0].set_ylabel(r"$\Delta\langle NN\rangle$ vs reference")
    for row in range(5):
        # Only where something was actually drawn, else matplotlib warns about an empty legend.
        if axes[row][0].get_legend_handles_labels()[0]:
            axes[row][0].legend(fontsize=8)

    fig.suptitle(f"Single-orbital Hubbard-Holstein, n={filling:g}\n"
                 "second estimator of the same quantity: dotted in tau, open markers in iw_n; "
                 "CTHYB lf=False is the analytic-vs-stochastic cross-check; "
                 "[...] marks a series that failed the sign / sum-rule / density check", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


drawn = 0
for filling in FILLINGS:
    fig = make_figure(filling)
    if fig is None:
        print(f"[n={filling:g}] no files under {DATA_DIR} for g={G:g}, omega_0={OMEGA_0:g} -- skipped")
        continue
    drawn += 1
    if SAVE_AS:
        name = SAVE_AS.format(g=G, filling=filling)
        fig.savefig(name, dpi=150, bbox_inches="tight")
        print(f"wrote {name}")
if not drawn:
    raise SystemExit(f"No files under {DATA_DIR} for g={G:g}, omega_0={OMEGA_0:g}.\n"
                     "Check DATA_DIR and the knobs at the top of this script.")
plt.show()
