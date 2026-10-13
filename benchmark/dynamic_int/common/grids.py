# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""The one tau grid and one Matsubara grid of every solver and ED script here, so curves
compare point by point. Each solver's correlator mesh is set from N_TAU:

  CTHYB  chiAB_tau on `n_tau`, Q_tau / D0_corr on `n_tau_bosonic`
  CTSEG  nn_tau on `n_tau_bosonic`
  CTINT  chiAB_tau on `n_tau_chi2` (library default 201)
  ED     `--n_tau`

N_TAU must be >= 2 N_IW + 1 for TRIQS' Fourier, and is odd so beta/2 is a mesh point. It is
well above that minimum because CTSEG interpolates Delta(tau) linearly on this mesh and CTHYB
takes its nearest point, which at beta = 100 needs the finer grid.

CTHYB runs in its DLR mode: Delta goes in, and G and Sigma come out, on the DLR Matsubara mesh
(beta, DLR_WMAX, DLR_EPS), whose frequencies are a subset of the regular grid's. DLR_WMAX is
CTINT's cutoff here, and G, not only Delta, must have no spectral weight above it.
"""
N_TAU = 4001
N_IW = 1025
DLR_WMAX = 10.0
DLR_EPS = 1e-10


def add_dlr_args(parser):
    parser.add_argument("--dlr_wmax", type=float, default=DLR_WMAX, help="DLR frequency cutoff of Delta, G and Sigma")
    parser.add_argument("--dlr_eps", type=float, default=DLR_EPS, help="DLR accuracy")
