# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""The one imaginary-time grid and one Matsubara grid every benchmark here uses.

Every tau quantity -- G(tau), the retarded kernels, and every correlator -- from every
solver and every ED script sits on the same N_TAU-point uniform grid on [0, beta], so two
runs compare point by point and no plot has to interpolate. Each solver has its own knob
for its correlator mesh, and all of them are set from N_TAU:

  CTHYB  O_tau is binned on `n_tau`, Q_tau / D0_corr on `n_tau_bosonic` -> both N_TAU
  CTSEG  nn_tau is on `n_tau_bosonic`                                    -> N_TAU
  CTINT  chiAB_tau is on `n_tau_chi2`, whose library default is 201      -> N_TAU
  ED     `--n_tau`                                                        -> N_TAU

Why 4001:
  * TRIQS' Fourier needs n_tau >= 2 n_iw + 1 = 2051 at N_IW = 1025.
  * CTHYB and CTSEG linearly interpolate Delta(tau) on the n_tau mesh. For the dmft bath
    at beta = 100 that costs 4.3e-4 of |Delta(0)| on 4001 points, the same as the 4096
    used before, against 1.6e-3 on 2051 -- as large as the solver differences being
    measured. At beta = 10 it is 4.5e-6.
  * Odd, so beta/2 is a mesh point.

The price is CTINT: its chiAB_tau insertion estimator costs time linear in the number of
points. At beta = 10 (16 ranks), going from 201 to 2001 points took the measurement from
14 s to 109 s and the throughput from 807 to 144 cycles/s.
"""
N_TAU = 4001
N_IW = 1025
