/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014, P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <algorithm>
#include <optional>
#include "../types.hpp"

// Pauli proposals of the pair moves (move_insert_c_cdag, move_remove_c_cdag), mixed with the uniform ones.
//
// With probability pauli_prob, the c is proposed with the inner index of its c^dagger, uniformly in the interval that keeps
// the operators of that index alternating: from the c^dagger to the nearest operator of the index on its right (earlier
// time) when that one is a c^dagger (an antisegment), else to the nearest one on its left (a segment). With no c^dagger or
// no c of the index, it goes anywhere on [0, beta). The Pauli removal of a c^dagger takes one of the nearest c of its index,
// on either side. For blocks of size 1 and a density-density h_loc, pauli_prob = 1 is the segment/antisegment
// insertion and removal of CT-SEG. The probabilities of both directions come from the same scan, so that they match.

namespace triqs_cthyb {

  /// The operators of one inner index of a block around a c^dagger at tau, d(t1, t2) = t1 - t2 being the cyclic distance
  struct pauli_scan_t {
    time_pt tau;
    int n_cdag = 0, n_c = 0;          // Number of c^dagger and c of the index
    time_pt cdag_R, c_R, cdag_L, c_L; // Distances d(tau, t) to the nearest c^dagger / c on the right, d(t, tau) on the left
    long c_R_col = -1, c_L_col = -1;  // Det columns of the nearest c on the right / left

    void add_cdag(time_pt t) {
      cdag_R = (n_cdag == 0 ? tau - t : std::min(cdag_R, tau - t));
      cdag_L = (n_cdag == 0 ? t - tau : std::min(cdag_L, t - tau));
      ++n_cdag;
    }

    void add_c(time_pt t, long col) {
      if (n_c == 0 or tau - t < c_R) {
        c_R     = tau - t;
        c_R_col = col;
      }
      if (n_c == 0 or t - tau < c_L) {
        c_L     = t - tau;
        c_L_col = col;
      }
      ++n_c;
    }

    /// Interval [start, start + length) of the Pauli c, if there are both a c^dagger and a c of the index
    struct gap_t {
      time_pt start, length;
      [[nodiscard]] bool contains(time_pt t) const { return t - start < length; }
    };
    [[nodiscard]] std::optional<gap_t> gap() const {
      if (n_cdag == 0 or n_c == 0) return {};
      if (cdag_R < c_R) return gap_t{tau - cdag_R, cdag_R};
      return gap_t{tau, std::min(cdag_L, c_L)};
    }

    /// Is the c in det column col one of the nearest c, i.e. a Pauli removal candidate (1 or 2 of them)?
    [[nodiscard]] bool is_nearest_c(long col) const { return n_c > 0 and (col == c_R_col or col == c_L_col); }
    [[nodiscard]] int n_nearest_c() const { return std::min(n_c, 2); }
  };

  /// Scan of the c^dagger and c of inner index a in the det (rows = c^dagger, columns = c), skipping one row and column
  template <typename Det> pauli_scan_t pauli_scan(Det const &det, int a, time_pt tau, long skip_row = -1, long skip_col = -1) {
    pauli_scan_t s{.tau = tau};
    for (long i = 0; i < det.size(); ++i) {
      if (auto const &[t, ai] = det.get_x(i); i != skip_row and ai == a) s.add_cdag(t);
      if (auto const &[t, ai] = det.get_y(i); i != skip_col and ai == a) s.add_c(t, i);
    }
    return s;
  }

  /// Probability density (per unit time and inner index) of proposing the c (index b at tau_c) to the c^dagger (index a)
  /// scanned in s, the configuration without the pair
  inline double pauli_insert_density(pauli_scan_t const &s, double pauli_prob, int a, int b, time_pt tau_c, int block_size, double beta) {
    double pauli = 0.0;
    if (a == b) {
      auto gap = s.gap();
      pauli    = (gap ? (gap->contains(tau_c) ? 1.0 / double(gap->length) : 0.0) : 1.0 / beta);
    }
    return pauli_prob * pauli + (1.0 - pauli_prob) / (block_size * beta);
  }

  /// Probability of choosing the c in det column col to remove with the c^dagger scanned in s, the configuration with the
  /// pair, which has k pairs in the block. Without a c of the index, the Pauli removal is uniform as well.
  inline double pauli_remove_prob(pauli_scan_t const &s, double pauli_prob, long col, long k) {
    double pauli = (s.n_c > 0 ? (s.is_nearest_c(col) ? 1.0 / s.n_nearest_c() : 0.0) : 1.0 / k);
    return pauli_prob * pauli + (1.0 - pauli_prob) / k;
  }

  /// Ratio of the proposal probabilities of the removal of the pair (c^dagger of index a at tau_cdag, c of index b at tau_c)
  /// and of its insertion into the det x of the configuration without it. Infinite if the insertion cannot propose it.
  template <typename Det>
  double pauli_insertion_ratio(Det const &x, double pauli_prob, int a, time_pt tau_cdag, int b, time_pt tau_c, int block_size, double beta) {
    long k     = x.size();
    auto scan  = pauli_scan(x, a, tau_cdag);
    double ins = pauli_insert_density(scan, pauli_prob, a, b, tau_c, block_size, beta);
    if (a == b) scan.add_c(tau_c, k); // Now the scan of the configuration with the pair, the c in column k
    double rem = pauli_remove_prob(scan, pauli_prob, k, k + 1);
    return (rem / (k + 1)) / (ins / (block_size * beta));
  }

  /// Ratio of the proposal probabilities of the insertion of the pair (row, col) of the det y and of its removal from y,
  /// i.e. the inverse of pauli_insertion_ratio for the configuration without the pair. Zero if the insertion cannot propose it.
  template <typename Det> double pauli_removal_ratio(Det const &y, double pauli_prob, long row, long col, int block_size, double beta) {
    auto const &[tau_cdag, a] = y.get_x(row);
    auto const &[tau_c, b]    = y.get_y(col);
    double rem                = pauli_remove_prob(pauli_scan(y, a, tau_cdag), pauli_prob, col, y.size());
    double ins                = pauli_insert_density(pauli_scan(y, a, tau_cdag, row, col), pauli_prob, a, b, tau_c, block_size, beta);
    return (ins / (block_size * beta)) / (rem / y.size());
  }

  /// A pair proposed for insertion: c^dagger of index a at tau_cdag, c of index b at tau_c
  struct pauli_pair_t {
    int a, b;
    time_pt tau_cdag, tau_c;
  };

  /// Propose a pair to insert into the det x, by the Pauli proposal with probability pauli_prob, uniformly otherwise.
  /// Empty if an operator of the index already sits at tau_cdag, where the insertion would fail anyway.
  template <typename Det, typename RNG>
  std::optional<pauli_pair_t> propose_pauli_insertion(Det const &x, double pauli_prob, int block_size, time_segment const &tau_seg, RNG &rng) {
    pauli_pair_t pair;
    pair.a        = rng(block_size);
    pair.tau_cdag = tau_seg.get_random_pt(rng);
    bool pauli    = rng() < pauli_prob;
    pair.b        = (pauli ? pair.a : rng(block_size));
    auto gap      = pauli_scan(x, pair.a, pair.tau_cdag).gap();
    if (pauli and gap) {
      if (double(gap->length) == 0.0) return {};
      pair.tau_c = gap->start + tau_seg.get_random_pt(rng, gap->length);
    } else
      pair.tau_c = tau_seg.get_random_pt(rng);
    return pair;
  }

  /// Propose a pair (row, col) to remove from the det y, by the Pauli removal with probability pauli_prob, uniformly otherwise
  template <typename Det, typename RNG> std::pair<long, long> propose_pauli_removal(Det const &y, double pauli_prob, RNG &rng) {
    long row                  = rng(y.size());
    auto const &[tau_cdag, a] = y.get_x(row);
    auto scan                 = pauli_scan(y, a, tau_cdag);
    bool pauli                = scan.n_c > 0 and rng() < pauli_prob;
    return {row, pauli ? (rng(2) == 0 ? scan.c_R_col : scan.c_L_col) : rng(y.size())};
  }

} // namespace triqs_cthyb
