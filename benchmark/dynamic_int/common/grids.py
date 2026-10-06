# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""The tau and Matsubara grids shared by every solver and ED script, so results compare point
by point. Each solver's correlator mesh is set from N_TAU: CTHYB n_tau and n_tau_bosonic, CTSEG
n_tau_bosonic, CTINT n_tau_chi2 (library default 201), ED --n_tau.

N_TAU = 4001: Fourier needs n_tau >= 2 N_IW + 1 = 2051, and the solvers' linear interpolation of
Delta(tau) costs 4.3e-4 of |Delta(0)| at beta = 100 on 4001 points against 1.6e-3 on 2051. Odd,
so beta/2 is a mesh point. CTINT's chiAB_tau estimator is linear in the number of points.
"""
N_TAU = 4001
N_IW = 1025
