# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
One h5 schema for every dynamic_int benchmark, so one plot script shape reads them all.

A result file always has

    solver        'cthyb' | 'ctseg' | 'ctint' | 'ed'
    params        dict of every command-line argument of the run
    beta, mu      the inputs, so a plot never has to guess which model was solved
    tau_G, G      G(tau) of the reference orbital, as plain arrays
    w_n           positive Matsubara frequencies
    Sigma         Sigma(iw_n) of the reference orbital on w_n (complex)

and, where the solver provides them,

    Sigma_alt     a second Sigma from an independent route, same mesh -- for cthyb the
                  G_tau Dyson inversion against the preferred Legendre one, for ctseg the
                  Dyson one against the improved estimator. The pair bounds the systematic.
    tau_corr, corr, corr_label   the benchmark's correlation function and its axis label
    corr_alt, corr_alt_label     a second estimator of it (e.g. cthyb's O_tau insertion
                  measurement vs the Legendre kink estimator)
    density       measured <n> per spin-orbital, the consistency check on the pinned mu
    average_sign, pert_order, pert_order_dyn
    ed_truncation the ED's phonon-truncation error, where applicable

plus a `raw` group holding whatever solver-specific objects are worth keeping (full
BlockGfs, K_n, Q_tau, ...). Plot scripts read only the flat arrays; `raw` is for when a
number looks wrong and the original object is needed.

Everything flat is a plain numpy array rather than a Gf, because that is what makes the
files readable without matching the writing solver's triqs build.
"""
import os

import numpy as np
from h5 import HDFArchive


def output_file(out_dir, benchmark, solver, beta, filling, tag=""):
    """`<out_dir>/<benchmark>_<solver>_b-<beta>_n-<filling>[_<tag>].h5`, directory created.

    `filling` is the *target* density per spin-orbital (0.5 at half filling), so a file
    name states the intended model and `density` inside states what was achieved.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = f"{benchmark}_{solver}_b-{beta:g}_n-{filling:g}"
    if tag:
        name += f"_{tag}"
    return os.path.join(out_dir, name + ".h5")


def _as_array(value):
    """Plain float/complex ndarray from a Gf, a histogram, or something already array-like."""
    if value is None:
        return None
    if hasattr(value, "data") and not isinstance(value, np.ndarray):
        value = value.data
    arr = np.asarray(value)
    return arr if np.iscomplexobj(arr) else arr.astype(float)


def save(filename, solver, params, beta, mu, tau_G, G, w_n=None, Sigma=None, Sigma_alt=None,
         tau_corr=None, corr=None, corr_label=None, corr_alt=None, corr_alt_label=None,
         density=None, average_sign=None, pert_order=None, pert_order_dyn=None,
         ed_truncation=None, raw=None):
    """Write one run to `filename` in the schema above. Optional fields are simply omitted."""
    with HDFArchive(filename, "w") as A:
        A["solver"] = solver
        A["params"] = {k: v for k, v in dict(params).items() if v is not None}
        A["beta"] = float(beta)
        A["mu"] = _as_array(mu)
        A["tau_G"] = _as_array(tau_G)
        A["G"] = _as_array(G)

        for key, value in (("w_n", w_n), ("Sigma", Sigma), ("Sigma_alt", Sigma_alt),
                           ("tau_corr", tau_corr), ("corr", corr), ("corr_alt", corr_alt),
                           ("density", density), ("pert_order", pert_order),
                           ("pert_order_dyn", pert_order_dyn)):
            if value is not None:
                A[key] = _as_array(value)

        for key, value in (("corr_label", corr_label), ("corr_alt_label", corr_alt_label)):
            if value is not None:
                A[key] = str(value)

        for key, value in (("average_sign", average_sign), ("ed_truncation", ed_truncation)):
            if value is not None:
                A[key] = float(np.real(value))

        if raw:
            A.create_group("raw")
            for key, value in raw.items():
                if value is not None:
                    A["raw"][key] = value
    print(f"Saved {filename}")


def load(filename, with_raw=False):
    """Read a result file into a dict, or None if it does not exist.

    Plot scripts call this and skip missing entries, so a partially finished grid still
    plots. `raw` is left out unless asked for, since it holds the large objects.
    """
    if not os.path.exists(filename):
        return None
    with HDFArchive(filename, "r") as A:
        out = {k: A[k] for k in A.keys() if k != "raw"}
        if with_raw and "raw" in A:
            out["raw"] = {k: A["raw"][k] for k in A["raw"].keys()}
    return out


def mean_order(histogram):
    """Mean perturbation order from a histogram array, or nan if absent."""
    if histogram is None:
        return float("nan")
    h = np.asarray(histogram, dtype=float)
    total = h.sum()
    return float((np.arange(len(h)) * h).sum() / total) if total > 0 else float("nan")
