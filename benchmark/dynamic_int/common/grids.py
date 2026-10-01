# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""The one tau grid and one Matsubara grid of every solver and ED script here, so curves
compare point by point. Each solver's correlator mesh is set from N_TAU:

  CTHYB  O_tau on `n_tau`, Q_tau / D0_corr on `n_tau_bosonic`
  CTSEG  nn_tau on `n_tau_bosonic`
  CTINT  chiAB_tau on `n_tau_chi2` (library default 201)
  ED     `--n_tau`

N_TAU must be >= 2 N_IW + 1 for TRIQS' Fourier, and is odd so beta/2 is a mesh point. It is
well above that minimum because CTHYB and CTSEG interpolate Delta(tau) linearly on this mesh,
which at beta = 100 needs the finer grid.
"""
N_TAU = 4001
N_IW = 1025
