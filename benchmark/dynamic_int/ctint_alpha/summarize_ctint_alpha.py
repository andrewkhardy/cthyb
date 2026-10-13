# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Summary of an alpha scan (run_ctint_alpha.sh), one table per model point, with every scheme against BASELINE:

  order      CTINT's average order, and its change from BASELINE (dk)
  pred       the change common/ctint.extra_order predicts from the shifts alone. Where dk and pred agree, the shifts
             cost exactly what <k> = -<S_int> says; a constant factor between them on D0 points means another D0
             convention
  sign, <n>  average sign and density per spin-orbital
  dG, dS     max |G(tau) - G_ref| and max |Sigma(iw_n) - Sigma_ref| over the first N_LOW frequencies (cluster: every
             diagonal entry). The references are the production runs: CTSEG for the impurity points, CTINT
             iteration 0 (signed alpha, the same Hartree-start G0) for the clusters. All runs had the same wall time,
             so a smaller error is a more efficient scheme, as long as it sits above the reference's own noise. A
             cluster dS for 'signed 0.51' well above the others' spread means the reference started from another G0
  asym       max |G_up(tau) - G_down(tau)|, a reference-free noise proxy (zero up to noise in the paramagnet; a lower
             bound, since both spins come from the same configurations)

    python summarize_ctint_alpha.py [scan_dir] [--ref_root DIR] [--plot FILE]

scan_dir holds hubbard_holstein/, spin_spin/ and cluster/; ref_root holds the production hubbard_holstein/,
spin_spin/ and cluster_hubbard/ (both default to the ceph data). Locally, e.g.
    python summarize_ctint_alpha.py ~/Documents/Data/ctint_alpha --ref_root ~/Documents/Data --plot alpha_scan.pdf
"""
import argparse
import glob
import os
import re

import numpy as np
import triqs.gfs  # noqa: F401  (registers the Gf h5 formats inside the cluster files' it0)
from h5 import HDFArchive

DATA = "/mnt/home/ahardy/ceph/CTHYB_Data"
FAMILIES = {  # scan subdirectory: (reference subdirectory, reference solvers in order of preference)
    "hubbard_holstein": ("hubbard_holstein", ["ctseg", "ctint"]),
    "spin_spin": ("spin_spin", ["ctseg", "ctint"]),
    "cluster": ("cluster_hubbard", ["ctint"]),
}
BASELINE = "signed 0.51"
SCHEME_ORDER = ["signed 0.51", "near 0.01", "centred 0.3", "centred 0.2", "centred 0.1", "centred 0.05", "library 0.1"]
N_LOW = 8
# Must match the reference; the cluster is in the file name (older runs stored it as L).
CLUSTER_PARAMS = ("scheme", "U", "t", "beta", "n_k", "n_iw", "n_tau", "dlr_wmax")
TAG = re.compile(r"^(?P<point>.+)_alpha-(?P<scheme>[a-z]+)-d(?P<delta>[0-9.]+(?:e-?[0-9]+)?)(?:-c(?P<center>[0-9.]+))?\.h5$")


KEYS = ("params", "alpha", "average_order", "average_sign", "density", "solve_seconds",
        "tau_G", "G", "G_down", "w_n", "Sigma", "it0")


def load(path):
    """The entries of `path` that the summary reads, skipping the large Gf objects."""
    with HDFArchive(path, "r") as A:
        return {k: A[k] for k in KEYS if k in A}


def max_diff(x, y, tau_x=None, tau_y=None):
    x, y = np.asarray(x), np.asarray(y)
    if tau_x is not None and (len(tau_x) != len(tau_y) or not np.allclose(tau_x, tau_y)):
        y = np.interp(tau_x, tau_y, y)
    return float(np.max(np.abs(x - y)))


def impurity_row(run, ref):
    row = dict(order=float(run["average_order"]), sign=float(np.real(run["average_sign"])),
               n=float(np.mean(run["density"])), time=float(run["solve_seconds"]),
               pred=run["alpha"].get("extra_order"),
               asym=max_diff(run["G"], run["G_down"]))
    if ref is not None:
        row["dG"] = max_diff(run["G"], ref["G"], run["tau_G"], ref["tau_G"])
        if np.allclose(run["w_n"][:N_LOW], ref["w_n"][:N_LOW]):
            row["dS"] = max_diff(run["Sigma"][:N_LOW], ref["Sigma"][:N_LOW])
    return row


def low_diagonal(sigma):
    """Diagonal of a cluster Sigma (n_w, N, N) over the full mesh, at the first N_LOW positive frequencies."""
    iw0 = sigma.shape[0] // 2
    return np.diagonal(sigma[iw0:iw0 + N_LOW], axis1=1, axis2=2)


def cluster_row(run, ref):
    it = run["it0"]
    row = dict(order=float(it["average_order"]), sign=float(it["average_sign"]), n=float(it["density"]) / 2,
               time=float(it["solve_seconds"]), pred=run["alpha"].get("extra_order"))
    if ref is not None:
        # The saved G0 is the next iteration's, so the parameters stand in for the starting G0.
        differ = [k for k in CLUSTER_PARAMS if run["params"].get(k) != ref["params"].get(k)]
        if differ:
            print(f"    (reference has other {', '.join(differ)}: no dS)")
        else:
            row["dS"] = max_diff(low_diagonal(it["Sigma"]), low_diagonal(ref["it0"]["Sigma"]))
    return row


def reference(ref_dir, point, solvers):
    """The first existing production run of `point` (a scan file name without its alpha tag), and its solver."""
    for solver in solvers:
        path = os.path.join(ref_dir, point.replace("_ctint", f"_{solver}", 1) + ".h5")
        if os.path.exists(path):
            return load(path), solver
    return None, None


def ordered(labels):
    """Scheme labels in SCHEME_ORDER, then any others."""
    return [s for s in SCHEME_ORDER if s in labels] + sorted(s for s in labels if s not in SCHEME_ORDER)


def point_label(point, family):
    """A scan point without its family prefix, e.g. 'b-10_n-0.5_g-0.7_w0-1'."""
    return point.replace("_ctint", "").removeprefix(family).strip("_")


def summarize(scan_dir, ref_root):
    """{family: {point: {scheme label: row}}}, printing one table per point."""
    out = {}
    for family, (ref_sub, solvers) in FAMILIES.items():
        points = {}
        for path in sorted(glob.glob(os.path.join(scan_dir, family, "*_alpha-*.h5"))):
            m = TAG.match(os.path.basename(path))
            if m:
                label = f"{m['scheme']} {float(m['delta']):g}" + (f" c{m['center']}" if m["center"] else "")
                points.setdefault(m["point"], {})[label] = path
        for point, files in points.items():
            ref, ref_solver = reference(os.path.join(ref_root, ref_sub), point, solvers)
            print(f"\n{point}   (reference: {ref_solver or 'none'})")
            rows = {}
            for label in ordered(files):
                run = load(files[label])
                rows[label] = cluster_row(run, ref) if family == "cluster" else impurity_row(run, ref)
            base = rows.get(BASELINE)
            print(f"  {'scheme':<15}{'order':>9}{'dk':>9}{'pred':>9}{'sign':>9}{'<n>':>9}"
                  f"{'dG':>10}{'dS':>10}{'asym':>10}{'time':>7}")
            for label, r in rows.items():
                dk = r["order"] - base["order"] if base else None
                pred = (r["pred"] - base["pred"] if base and r["pred"] is not None and base["pred"] is not None
                        else None)
                print(f"  {label:<15}{r['order']:9.1f}{_f(dk, '9.1f')}{_f(pred, '9.1f')}{r['sign']:9.4f}{r['n']:9.4f}"
                      f"{_f(r.get('dG'), '10.2e')}{_f(r.get('dS'), '10.2e')}{_f(r.get('asym'), '10.2e')}{r['time']:7.0f}")
            out.setdefault(family, {})[point] = rows
    return out


def _f(x, fmt):
    width = int(fmt.split(".")[0])
    return f"{x:{fmt}}" if x is not None else f"{'-':>{width}}"


# Categorical slots in fixed order, one per point, with a marker each so identity is not color alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot(results, path):
    """Columns: families; rows: order and Sigma error relative to BASELINE, and the sign; x: the schemes."""
    import matplotlib.pyplot as plt

    families = [f for f in FAMILIES if f in results]
    fig, axes = plt.subplots(3, len(families), figsize=(5.2 * len(families), 9.0), squeeze=False, sharex="col")
    for col, family in enumerate(families):
        points = results[family]
        labels = ordered({s for rows in points.values() for s in rows})
        x_of = {s: i for i, s in enumerate(labels)}
        for k, (point, rows) in enumerate(points.items()):
            base = rows.get(BASELINE)
            style = dict(color=COLORS[k % len(COLORS)], marker=MARKERS[k % len(MARKERS)], markersize=6,
                         linestyle="none", label=point_label(point, family))
            xs = [x_of[s] + 0.08 * (k - (len(points) - 1) / 2) for s in labels if s in rows]
            present = [s for s in labels if s in rows]
            if base:
                axes[0][col].plot(xs, [rows[s]["order"] / base["order"] for s in present], **style)
                if base.get("dS"):
                    axes[2][col].plot(xs, [rows[s].get("dS", np.nan) / base["dS"] for s in present], **style)
            axes[1][col].plot(xs, [rows[s]["sign"] for s in present], **style)
        axes[0][col].set_title(family)
        axes[0][col].set_ylabel(r"$\langle k\rangle / \langle k\rangle_{\mathrm{signed\ 0.51}}$")
        axes[1][col].set_ylabel("average sign")
        axes[2][col].set_ylabel(r"$|\Delta\Sigma|_{\mathrm{low}}$ / signed 0.51")
        axes[2][col].set_yscale("log")
        axes[2][col].set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        for row, ref_line in ((0, 1.0), (1, 1.0), (2, 1.0)):
            ax = axes[row][col]
            ax.axhline(ref_line, color="#9a9a96", linewidth=0.8, zorder=0)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0][col].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scan_dir", nargs="?", default=os.path.join(DATA, "ctint_alpha"))
    parser.add_argument("--ref_root", default=DATA, help="Directory with the production runs' subdirectories")
    parser.add_argument("--plot", default=None, help="Save the comparison figure here")
    args = parser.parse_args()
    results = summarize(os.path.expanduser(args.scan_dir), os.path.expanduser(args.ref_root))
    if args.plot and results:
        plot(results, args.plot)
