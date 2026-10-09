// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

// The proposal probabilities of the Pauli pair moves (moves/pauli.hpp), on random configurations of a mock det with
// blocks of size 1 and 2, operators that alternate for each inner index or not, and pauli_prob = 0, 0.3, 1:
//   - the insertion and removal ratios of the same pair are inverse of each other, from either side,
//   - the proposals follow the probabilities the ratios assume: E[1_R / q] = |R| over the proposed insertions, and the
//     frequencies of the proposed removals,
//   - pauli_prob = 0 gives the uniform ratios, and pauli_prob = 1 keeps alternating operators alternating.

#include <triqs/test_tools/arrays.hpp>
#include <triqs/mc_tools/random_generator.hpp>
#include <triqs_cthyb/moves/pauli.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

using namespace triqs_cthyb;

namespace {

  double const beta = 10.0;

  // The (time, inner index) of the c^dagger (rows) and c (columns) of a block
  struct mock_det {
    std::vector<op_t> x, y;
    [[nodiscard]] long size() const { return static_cast<long>(x.size()); }
    [[nodiscard]] op_t const &get_x(long i) const { return x[i]; }
    [[nodiscard]] op_t const &get_y(long j) const { return y[j]; }
  };

  using rng_t = triqs::mc_tools::random_generator;

  // k pairs of a block of size n: for each inner index, an even number of times taken alternately by c^dagger and c
  // (alternating), or every operator at a random time with a random index
  mock_det random_det(rng_t &rng, time_segment const &tau_seg, int n, int k, bool alternating) {
    mock_det det;
    if (not alternating) {
      for (int i = 0; i < k; ++i) {
        det.x.emplace_back(tau_seg.get_random_pt(rng), rng(n));
        det.y.emplace_back(tau_seg.get_random_pt(rng), rng(n));
      }
      return det;
    }
    std::vector<int> pairs(n, 0);
    for (int i = 0; i < k; ++i) ++pairs[rng(n)];
    for (int a = 0; a < n; ++a) {
      std::vector<time_pt> times;
      for (int i = 0; i < 2 * pairs[a]; ++i) times.push_back(tau_seg.get_random_pt(rng));
      std::ranges::sort(times);
      int first_dagger = rng(2);
      for (int i = 0; i < 2 * pairs[a]; ++i) ((i % 2 == first_dagger) ? det.x : det.y).emplace_back(times[i], a);
    }
    return det;
  }

  mock_det with_pair(mock_det det, pauli_pair_t const &pair) {
    det.x.emplace_back(pair.tau_cdag, pair.a);
    det.y.emplace_back(pair.tau_c, pair.b);
    return det;
  }

  mock_det without_pair(mock_det det, long row, long col) {
    det.x.erase(det.x.begin() + row);
    det.y.erase(det.y.begin() + col);
    return det;
  }

  // Do the operators of every inner index alternate between c^dagger and c around the circle?
  bool alternates(mock_det const &det) {
    std::map<int, std::vector<std::pair<time_pt, bool>>> ops;
    for (long i = 0; i < det.size(); ++i) {
      ops[det.get_x(i).second].emplace_back(det.get_x(i).first, true);
      ops[det.get_y(i).second].emplace_back(det.get_y(i).first, false);
    }
    for (auto &[a, v] : ops) {
      std::ranges::sort(v);
      for (size_t i = 0; i < v.size(); ++i)
        if (v[i].second == v[(i + 1) % v.size()].second) return false;
    }
    return true;
  }

  struct setup_t {
    int n;
    bool alternating;
    double p;
  };
  std::vector<setup_t> const setups = {{1, true, 0.0}, {1, true, 0.3}, {1, true, 1.0}, {2, true, 0.3},
                                       {2, true, 1.0}, {2, false, 0.0}, {2, false, 0.3}, {2, false, 1.0}};

} // namespace

// Insertion ratio x -> y times removal ratio y -> x is 1, or both directions are impossible together
TEST(PauliProposal, InverseRatios) {
  rng_t rng("mt19937", 2718);
  time_segment tau_seg{beta};
  int n_checked = 0;
  for (auto const &[n, alternating, p] : setups) {
    for (int trial = 0; trial < 4000; ++trial) {
      int k = rng(7);

      // From the insertion side
      auto x    = random_det(rng, tau_seg, n, k, alternating);
      auto pair = propose_pauli_insertion(x, p, n, tau_seg, rng);
      if (pair) {
        double r_ins = pauli_insertion_ratio(x, p, pair->a, pair->tau_cdag, pair->b, pair->tau_c, n, beta);
        double r_rem = pauli_removal_ratio(with_pair(x, *pair), p, k, k, n, beta);
        if (r_ins == 0.0)
          EXPECT_TRUE(std::isinf(r_rem));
        else
          EXPECT_NEAR(r_ins * r_rem, 1.0, 1e-12);
        ++n_checked;
      }

      // From the removal side
      auto y = random_det(rng, tau_seg, n, k + 1, alternating);
      auto [row, col] = propose_pauli_removal(y, p, rng);
      double r_rem    = pauli_removal_ratio(y, p, row, col, n, beta);
      auto const &[tau_cdag, a] = y.get_x(row);
      auto const &[tau_c, b]    = y.get_y(col);
      double r_ins              = pauli_insertion_ratio(without_pair(y, row, col), p, a, tau_cdag, b, tau_c, n, beta);
      if (r_rem == 0.0)
        EXPECT_TRUE(std::isinf(r_ins));
      else
        EXPECT_NEAR(r_ins * r_rem, 1.0, 1e-12);
    }
  }
  EXPECT_GT(n_checked, 30000);
}

// pauli_prob = 0 reproduces the uniform ratios (n beta / (k + 1))^2 and (k / (n beta))^2
TEST(PauliProposal, UniformLimit) {
  rng_t rng("mt19937", 31415);
  time_segment tau_seg{beta};
  for (int n : {1, 2})
    for (int trial = 0; trial < 1000; ++trial) {
      int k     = rng(7);
      auto x    = random_det(rng, tau_seg, n, k, trial % 2 == 0);
      auto pair = propose_pauli_insertion(x, 0.0, n, tau_seg, rng);
      ASSERT_TRUE(pair);
      EXPECT_NEAR(pauli_insertion_ratio(x, 0.0, pair->a, pair->tau_cdag, pair->b, pair->tau_c, n, beta), std::pow(n * beta / (k + 1), 2),
                  1e-10);
      EXPECT_NEAR(pauli_removal_ratio(with_pair(x, *pair), 0.0, k, k, n, beta), std::pow((k + 1) / (n * beta), 2), 1e-12);
    }
}

// With pauli_prob = 1, an insertion into alternating operators keeps them alternating (no proposal of zero trace for a
// density-density h_loc), and so does a removal
TEST(PauliProposal, KeepsAlternation) {
  rng_t rng("mt19937", 1618);
  time_segment tau_seg{beta};
  for (int n : {1, 2})
    for (int trial = 0; trial < 4000; ++trial) {
      int k  = rng(7);
      auto x = random_det(rng, tau_seg, n, k, true);
      if (auto pair = propose_pauli_insertion(x, 1.0, n, tau_seg, rng)) EXPECT_TRUE(alternates(with_pair(x, *pair)));
      auto y          = random_det(rng, tau_seg, n, k + 1, true);
      auto [row, col] = propose_pauli_removal(y, 1.0, rng);
      EXPECT_TRUE(alternates(without_pair(y, row, col)));
    }
}

// The proposed insertions have the density q the ratios assume: E[1_R(z) / q(z)] = |R| for regions R of
// (a, tau_cdag, b, tau_c) with |R| counted in units of time^2. The regions split tau_c - tau_cdag into bins and a == b or not.
TEST(PauliProposal, InsertionDensity) {
  rng_t rng("mt19937", 1414);
  time_segment tau_seg{beta};
  int const n_bins = 8, n_samples = 400000;
  for (auto const &[n, alternating, p] : setups) {
    if (p == 1.0) continue; // q vanishes on part of the space, where E[1_R / q] only covers the support
    for (int config = 0; config < 3; ++config) {
      auto x = random_det(rng, tau_seg, n, 2 + config, alternating);
      std::vector<double> sum(2 * n_bins, 0.0);
      for (int s = 0; s < n_samples; ++s) {
        auto pair = propose_pauli_insertion(x, p, n, tau_seg, rng);
        ASSERT_TRUE(pair);
        double q = pauli_insert_density(pauli_scan(x, pair->a, pair->tau_cdag), p, pair->a, pair->b, pair->tau_c, n, beta) / (n * beta);
        int bin  = std::min(n_bins - 1, static_cast<int>(double(pair->tau_c - pair->tau_cdag) / beta * n_bins));
        sum[(pair->a == pair->b ? 0 : n_bins) + bin] += 1.0 / q;
      }
      for (int r = 0; r < 2 * n_bins; ++r) {
        double expected = (r < n_bins ? n : n * (n - 1)) * beta * beta / n_bins;
        EXPECT_NEAR(sum[r] / n_samples, expected, 0.03 * n * n * beta * beta / n_bins)
           << "n = " << n << ", alternating = " << alternating << ", p = " << p << ", region " << r;
      }
    }
  }
}

// The proposed removals have the probabilities the ratios assume: (1 / k) pauli_remove_prob for each (row, col)
TEST(PauliProposal, RemovalFrequencies) {
  rng_t rng("mt19937", 1732);
  time_segment tau_seg{beta};
  int const n_samples = 200000;
  for (auto const &[n, alternating, p] : setups)
    for (int config = 0; config < 3; ++config) {
      auto y = random_det(rng, tau_seg, n, 3 + config, alternating);
      long k = y.size();
      std::vector<double> count(k * k, 0.0);
      for (int s = 0; s < n_samples; ++s) {
        auto [row, col] = propose_pauli_removal(y, p, rng);
        count[row * k + col] += 1.0;
      }
      for (long row = 0; row < k; ++row)
        for (long col = 0; col < k; ++col) {
          double prob  = pauli_remove_prob(pauli_scan(y, y.get_x(row).second, y.get_x(row).first), p, col, k) / k;
          double sigma = std::sqrt(prob * (1 - prob) / n_samples);
          EXPECT_NEAR(count[row * k + col] / n_samples, prob, 5 * sigma + 1e-12)
             << "n = " << n << ", alternating = " << alternating << ", p = " << p << ", (" << row << ", " << col << ")";
        }
    }
}

MAKE_MAIN;
