# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Plots of the cluster_cthyb.py (CTHYB) and cluster_ctint.py (CTINT) runs in a data directory, two figures per cluster:

  iterations   rows DCA / CDMFT; columns average sign, expansion order, autocorrelation time (CTHYB) and the change of
               Sigma between successive iterations, max over K and 0 <= w_n <= --w_max of |Sigma_it - Sigma_it-1|.
  sigma        rows Im / Re Sigma(K, iw_n) for DCA and CDMFT; one column per star of K under the point group.
               Markers: the first iteration, where every solver starts from the same Hartree G0, so the solvers
               compare directly; lines: the last iteration of runs with more than one.

Sigma(K) is the diagonal of the cluster Fourier transform: from the cos/sin K blocks of CTHYB (the Sigma_iw the loop
went on with: by CRM on the DLR frequencies, tail-fitted on the regular mesh in earlier runs), from the site-basis
matrix of CTINT (on its DLR frequencies, on the regular mesh in earlier runs) and of CTHYB --basis sites (labelled
"sites"); from Sigma_sites, the site-basis Sigma of CTHYB, in every basis of the runs that save it. Each run is drawn on the frequencies of its own mesh. It
is averaged over the spins and over the K of a star, which assumes the full point group of the square. Both the --L
files of the earlier drivers and the --cluster files of the current ones are read. A summary table of every run goes to
stdout, with the cost of the last CTHYB solve where the file has it: measurements per second and rank, independent
samples per second and rank (sign^2 / (1 + 2 tau) of them), the seconds outside the Monte Carlo (atomic problem and
post-processing) and the invariant subspaces of h_loc.

    python plot_cluster.py --data_dir ~/Documents/Data/cluster_hubbard --save "cluster_{cluster}_{figure}.pdf"
"""
import argparse
import glob
import os
import re
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from triqs.gfs import Gf  # also registers Gf and BlockGf with h5
from h5 import HDFArchive

from clusters import Cluster

DATA_DIRS = ("~/Documents/Data/cluster_hubbard", "/mnt/home/ahardy/ceph/CTHYB_Data/cluster_hubbard")  # the first that exists

parser = argparse.ArgumentParser(description="Plot the cluster DCA / CDMFT runs of cluster_cthyb.py and cluster_ctint.py")
parser.add_argument("--data_dir", default=next((d for d in DATA_DIRS if os.path.isdir(os.path.expanduser(d))), DATA_DIRS[-1]))
parser.add_argument("--w_max", type=float, default=10.0, help="Largest Matsubara frequency plotted")
parser.add_argument("--save", default=None, help='Format for the files, e.g. "cluster_{cluster}_{figure}.pdf"; show if absent')
args, _ = parser.parse_known_args()  # known only: a Jupyter kernel passes its own --f=...

FILE = re.compile(r"cluster_(?P<cluster>[^_]+)_(?P<scheme>DCA|CDMFT)_U(?P<U>[\d.]+)_beta(?P<beta>[\d.]+)_mu(?P<mu>[-\d.]+)"
                  r"_(?P<tag>ctint|pauli[\d.]+(?:_(?:crm|dyson)_eps[\de.+-]+)?(?:_(?:waves|irreps|sites))?)\.h5$")
SCHEMES = ("DCA", "CDMFT")
SPINS = ("up", "dn")

COLOR = {"CTINT": "#2a78d6", "CTHYB p=0": "#eb6834", "CTHYB p=0.5": "#1baf7a"}  # slots 1-3 of the dataviz palette
MARKER = {"CTINT": "o", "CTHYB p=0": "s", "CTHYB p=0.5": "^"}
MORE = (("#eda100", "D"), ("#e87ba4", "v"), ("#008300", "P"), ("#6250d6", "X"), ("#e34948", "*"))  # slots 4-8, for the rest
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MIN_SPAN = 0.05  # smallest y range of a Sigma panel


def label_of(tag):
    """CTINT; CTHYB p=0.5 for the regular-mesh runs; CTHYB p=0.5 CRM ε=1e-07 for the DLR ones (Sigma fed back, DLR eps),
    with " sites" or " irreps" for those bases"""
    if tag == "ctint":
        return "CTINT"
    m = re.fullmatch(r"pauli([\d.]+)(?:_(crm|dyson)_eps([^_]+))?(?:_(waves|irreps|sites))?", tag)
    label = f"CTHYB p={float(m[1]):g}"
    if m[2] is not None:
        label += f" {'CRM' if m[2] == 'crm' else 'Dyson'} ε={float(m[3]):.0e}"
    return label + (f" {m[4]}" if m[4] in ("irreps", "sites") else "")


# ---- Cluster momenta of a run, in either file format ----

def sites_and_momenta(params):
    """R and K of the run: the --L drivers took R = (x, y) row by row and K = 2 pi R / L; the --cluster ones clusters.py"""
    if "L" in params:
        L = int(params["L"])
        sites = np.array([[x, y] for y in range(L) for x in range(L)])
        return sites, 2 * np.pi / L * sites
    cl = Cluster(params["cluster"], 1)
    return cl.sites, cl.K


def momentum_of_block(name, L):
    """K of a CTHYB block: 'K10c' (--L drivers, K = 2 pi (m, n) / L) or 'K+4_+0s' (clusters.py, in units of pi/4)"""
    old = re.fullmatch(r"K(\d)(\d)[cs]?", name)
    if old:
        return 2 * np.pi / L * np.array([int(old[1]), int(old[2])])
    new = re.fullmatch(r"K([+-]\d+)_([+-]\d+)[cs]?", name)
    return np.pi / 4 * np.array([int(new[1]), int(new[2])])


def star(K):
    """The representative (kx, ky), pi >= kx >= ky >= 0, of the orbit of K under the point group of the square"""
    a = np.abs((np.asarray(K) + np.pi) % (2 * np.pi) - np.pi)
    a[np.isclose(a, 0)] = 0  # wrap leaves -pi for pi: |.| fixes that
    return tuple(np.round(sorted(a, reverse=True), 9))


def star_name(s):
    def part(k):
        f = Fraction(k / np.pi).limit_denominator(8)
        return "0" if f == 0 else ("π" if f == 1 else (f"{f.numerator}π/{f.denominator}" if f.numerator != 1 else f"π/{f.denominator}"))
    return f"({part(s[0])}, {part(s[1])})"


# ---- Loading ----

def add_by_star(acc, sigma_site, params):
    """Append the diagonal of the cluster Fourier transform of a site-basis Sigma, (n_w, N, N), to acc by star"""
    sites, K = sites_and_momenta(params)
    U = np.exp(1j * sites @ K.T) / np.sqrt(len(sites))  # [R, K]
    sigma_K = np.einsum("rk,wrs,sk->wk", U.conj(), sigma_site, U)
    for k, Kv in enumerate(K):
        acc.setdefault(star(Kv), []).append(sigma_K[:, k])


def sigma_by_star(it_group, params):
    """`(w_n, {star: Sigma(K, iw_n)})` on the positive frequencies of the run's Matsubara mesh (regular, or DLR), averaged
    over spins and the K of each star"""
    acc = {}
    sigma = it_group["Sigma_iw"] if "Sigma_iw" in it_group else None
    if "Sigma_sites" in it_group:  # CTHYB, in any basis, from its Sigma in the site basis
        sigma = it_group["Sigma_sites"]
        w = np.array([complex(x).imag for x in sigma.mesh.values()])
        add_by_star(acc, sigma.data, params)
    elif "Sigma" in it_group:  # CTINT before the DLR loop, (n_w, N, N) in the site basis, on the regular mesh
        add_by_star(acc, it_group["Sigma"], params)
        n_w = len(it_group["Sigma"])
        w = (2 * np.arange(-(n_w // 2), n_w - n_w // 2) + 1) * np.pi / float(params["beta"])
    elif isinstance(sigma, Gf):  # CTINT, N x N in the site basis, on its DLR mesh
        w = np.array([complex(x).imag for x in sigma.mesh.values()])
        add_by_star(acc, sigma.data, params)
    elif params.get("basis") == "sites":  # CTHYB in the site basis, one N x N block per spin
        for name, g in sigma:
            w = np.array([complex(x).imag for x in g.mesh.values()])
            add_by_star(acc, g.data, params)
    else:  # CTHYB, blocks of the cos/sin waves joined by "|" (one wave per block in DCA), per spin
        L = int(params.get("L", 0))
        for name, g in sigma:
            w = np.array([complex(x).imag for x in g.mesh.values()])
            for i, wave in enumerate(name.split("_", 1)[1].split("|")):
                acc.setdefault(star(momentum_of_block(wave, L)), []).append(g.data[:, i, i])
    pos = w > 0
    return w[pos], {s: np.mean(v, axis=0)[pos] for s, v in sorted(acc.items())}


def load(path):
    """The run in `path`, or None while it has no finished iteration"""
    with HDFArchive(path, "r") as A:
        if "n_iter" not in A:
            return None
        params = A["params"]
        its = []
        for i in range(A["n_iter"]):
            g = A[f"it{i}"]
            wn, sigma = sigma_by_star(g, params)
            its.append(dict(sigma=sigma, **{k: g[k] for k in ("density", "average_sign", "average_order", "solve_seconds",
                                                              "auto_corr_time", "n_measures", "warmup_seconds",
                                                              "accumulation_seconds", "n_ranks", "n_subspaces",
                                                              "largest_subspace")
                                           if k in g}))
    keep = wn <= args.w_max
    for it in its:
        it["sigma"] = {s: v[keep] for s, v in it["sigma"].items()}
    return dict(params=params, wn=wn[keep], its=its)


runs = {}  # (cluster, U, beta, mu) -> scheme -> label -> run
for path in sorted(glob.glob(os.path.join(os.path.expanduser(args.data_dir), "cluster_*.h5"))):
    m = FILE.search(os.path.basename(path))
    if not m:
        continue
    try:
        run = load(path)
    except (OSError, KeyError, RuntimeError) as e:  # synced while the run was writing it
        print(f"Skipping {os.path.basename(path)}: unreadable ({e})")
        continue
    if run is None:
        print(f"Skipping {os.path.basename(path)}: no finished iteration yet")
        continue
    key = (m["cluster"], m["U"], m["beta"], m["mu"])
    runs.setdefault(key, {}).setdefault(m["scheme"], {})[label_of(m["tag"])] = run
if not runs:
    print(f"No cluster_*.h5 runs in {args.data_dir}")
extra = sorted({label for schemes in runs.values() for by_label in schemes.values() for label in by_label} - set(COLOR))
if len(extra) > len(MORE):
    raise SystemExit(f"{len(extra)} kinds of CTHYB run in {args.data_dir}, more than the {len(MORE)} colors left: move some aside")
for label, (color, marker) in zip(extra, MORE):
    COLOR[label], MARKER[label] = color, marker

# ---- Summary ----

def cost_columns(it):
    """Measurements and independent samples per second and rank, seconds outside the Monte Carlo, subspaces"""
    if "n_measures" not in it:
        return f"{'':>9s} {'':>9s} {'':>8s} {'':>12s}"
    rate = it["n_measures"] / (it["n_ranks"] * it["accumulation_seconds"])
    indep = rate * it["average_sign"] ** 2 / (1 + 2 * it["auto_corr_time"])
    other = it["solve_seconds"] - it["warmup_seconds"] - it["accumulation_seconds"]
    subspaces = f"{it['n_subspaces']}/{it['largest_subspace']}"
    return f"{rate:9.3g} {indep:9.3g} {other:8.1f} {subspaces:>12s}"


print(f"{'run':66s} {'iter':>4s} {'<n>':>7s} {'sign':>7s} {'order':>7s} {'autocorr':>8s} {'solve s':>8s} "
      f"{'meas/s':>9s} {'indep/s':>9s} {'other s':>8s} {'subspaces':>12s}")
for (cluster, U, beta, mu), schemes in runs.items():
    for scheme, by_label in schemes.items():
        for label, run in sorted(by_label.items()):
            last = run["its"][-1]
            print(f"{f'{cluster} {scheme} U={U} beta={beta} {label}':66s} {len(run['its']):4d} {last['density']:7.4f} "
                  f"{last['average_sign']:7.4f} {last['average_order']:7.1f} {last.get('auto_corr_time', np.nan):8.2f} "
                  f"{last['solve_seconds']:8.0f} {cost_columns(last)}")

# ---- Figures ----


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def finish(fig, cluster, figure):
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save.format(cluster=cluster, figure=figure), dpi=150)
        plt.close(fig)


def iteration_figure(key, schemes):
    cluster, U, beta, mu = key
    cols = ("average sign", "expansion order", "autocorrelation time (cycles)", r"max $|\Sigma_{it} - \Sigma_{it-1}|$")
    fig, axes = plt.subplots(len(SCHEMES), len(cols), figsize=(4 * len(cols), 3.2 * len(SCHEMES)), squeeze=False)
    for row, scheme in enumerate(SCHEMES):
        for label, run in sorted(schemes.get(scheme, {}).items()):
            its = run["its"]
            n = np.arange(len(its))
            kw = dict(color=COLOR[label], marker=MARKER[label], markersize=5.5, linewidth=2, label=label)
            axes[row, 0].plot(n, [it["average_sign"] for it in its], **kw)
            axes[row, 1].plot(n, [it["average_order"] for it in its], **kw)
            if "auto_corr_time" in its[0]:
                axes[row, 2].plot(n, [it["auto_corr_time"] for it in its], **kw)
            if len(its) > 1:
                change = [max(np.abs(its[i]["sigma"][s] - its[i - 1]["sigma"][s]).max() for s in its[i]["sigma"])
                          for i in range(1, len(its))]
                axes[row, 3].plot(n[1:], change, **kw)
        for col, title in enumerate(cols):
            ax = axes[row, col]
            style(ax)
            ax.set_title(f"{scheme}: {title}", fontsize=10, color=INK)
            ax.set_xlabel("iteration", fontsize=9, color=INK)
            ax.xaxis.get_major_locator().set_params(integer=True)
        axes[row, 1].set_yscale("log")
        axes[row, 3].set_yscale("log")
        axes[row, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cluster}, U = {U}, β = {beta}, μ = {mu}: per iteration", fontsize=11, color=INK)
    finish(fig, cluster, "iterations")


def sigma_figure(key, schemes):
    cluster, U, beta, mu = key
    stars = sorted({s for by_label in schemes.values() for run in by_label.values() for s in run["its"][0]["sigma"]})
    rows = [(scheme, part) for scheme in SCHEMES if scheme in schemes for part in ("Im", "Re")]
    fig, axes = plt.subplots(len(rows), len(stars), figsize=(3.4 * len(stars), 2.7 * len(rows)), squeeze=False, sharex=True)
    for r, (scheme, part) in enumerate(rows):
        take = np.imag if part == "Im" else np.real
        for label, run in sorted(schemes[scheme].items()):
            first, last = run["its"][0], run["its"][-1]
            for c, s in enumerate(stars):
                ax = axes[r, c]
                ax.plot(run["wn"], take(first["sigma"][s]), linestyle="none", marker=MARKER[label], markersize=5.5,
                        markerfacecolor="none", markeredgewidth=1.1, color=COLOR[label], label=f"{label}, it 0")
                if len(run["its"]) > 1:
                    ax.plot(run["wn"], take(last["sigma"][s]), linewidth=2, color=COLOR[label],
                            label=f"{label}, it {len(run['its']) - 1}")
        for c, s in enumerate(stars):
            ax = axes[r, c]
            style(ax)
            lo, hi = ax.get_ylim()  # Re Sigma(K) = U/2 by particle-hole symmetry at some K: keep its noise to scale
            if hi - lo < MIN_SPAN:
                ax.set_ylim((lo + hi - MIN_SPAN) / 2, (lo + hi + MIN_SPAN) / 2)
            ax.ticklabel_format(axis="y", useOffset=False)
            if part == "Re":
                ax.axhline(float(U) / 2, color=MUTED, linewidth=0.8, linestyle="--")
            ax.set_title(f"{scheme}, K = {star_name(s)}", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(rf"{part} $\Sigma(K, i\omega_n)$", fontsize=9, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel(r"$\omega_n$", fontsize=9, color=INK)
        axes[r, -1].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"{cluster}, U = {U}, β = {beta}, μ = {mu}: self-energy (dashed: U/2)", fontsize=11, color=INK)
    finish(fig, cluster, "sigma")


for key, schemes in runs.items():
    if any(len(run["its"]) > 1 for by_label in schemes.values() for run in by_label.values()):
        iteration_figure(key, schemes)
    sigma_figure(key, schemes)

if not args.save:
    plt.show()
