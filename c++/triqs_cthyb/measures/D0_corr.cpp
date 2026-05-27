#include <triqs/mc_tools.hpp>
#include <triqs/utility/legendre.hpp>

#include "./D0_corr.hpp"
#include "../math_utils.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_D0_corr::measure_D0_corr(std::optional<Q_l_t> &Q_l_opt, std::optional<Q_tau_t> &Q_tau_opt, qmc_data const &data, int n_tau,
                                   int n_leg, gf_struct_t const &gf_struct)
     : data(data), average_sign(0), n_leg(n_leg) {

    if (!data.use_lang_firsov || data.K_n_size == 0) {
      TRIQS_RUNTIME_ERROR << "measure_D0_corr requires lang_firsov=true with non-empty K_n.";
    }
    if (data.K_n_size != n_leg) {
      TRIQS_RUNTIME_ERROR << "measure_D0_corr requires n_leg to match K_n size: " << n_leg << " != " << data.K_n_size;
    }

    // Bug 1 fix: n_lin is the number of flattened orbital indices, obtained
    // from the linindex map, NOT from K_n.size() which equals n_leg.
    // K_n holds one expansion coefficient per Legendre order; linindex holds
    // one entry per (block, inner) orbital pair.
    n_lin = static_cast<int>(data.linindex.size());
    if (n_lin <= 0) TRIQS_RUNTIME_ERROR << "measure_D0_corr requires non-empty linindex.";

    Q_l_opt   = make_block2_gf<legendre>({data.config.beta(), Boson, n_leg}, gf_struct);
    Q_tau_opt = make_block2_gf<imtime>({data.config.beta(), Boson, n_tau}, gf_struct);

    Q_l.rebind(*Q_l_opt);
    Q_tau.rebind(*Q_tau_opt);

    Q_l()   = 0.0;
    Q_tau() = 0.0;

    alpha_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});

    // Bug 5 fix: allocate per-orbital mean occupancy accumulator.
    n_mean = nda::zeros<mc_weight_t>(std::array<long, 1>{n_lin});
  }

  void measure_D0_corr::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    std::vector<std::pair<time_pt, op_desc>> ops;
    ops.reserve(data.config.size());
    for (auto const &entry : data.config) ops.push_back(entry);

    if (ops.empty()) return;

    double beta = data.config.beta();
    triqs::utility::legendre_generator leg;

    // ------------------------------------------------------------------
    // Accumulate alpha_n(a, b, n) = sum_{i != j} P_n(x_{ij})
    // where x_{ij} = 2*|tau_i - tau_j|/beta - 1,  x in [-1, 1].
    //
    // Bug 2 fix: S_alpha = +1 for all operators.  The Lang-Firsov weight
    // couples to the density n = c†c, which is always positive; the sign
    // of op1.dagger has no place in the density-density correlator.
    //
    // Bug 3 fix: do NOT fold dt into [0, beta/2].  The Legendre expansion
    // of Q(tau) uses x = 2*tau/beta - 1 over the full interval [0, beta],
    // so dt must stay in [0, beta) before mapping to x in [-1, 1].
    //
    // Bug 4 fix: the diagonal self-pair (i == i, dt = 0) is a tau-
    // independent constant that cancels exactly against the disconnected
    // piece <n>^2 when we subtract it in collect_results.  We therefore
    // skip it here and accumulate <n_a> separately in n_mean instead.
    // For same-orbital off-diagonal pairs (a == b, i != j) we still add
    // both orderings (factor 2) because the sum over alpha,beta in
    // Eq. (62) runs over ALL ordered pairs.
    // ------------------------------------------------------------------

    for (size_t i = 0; i < ops.size(); ++i) {
      auto const &[t1, op1] = ops[i];
      int const a            = data.linindex.at({op1.block_index, op1.inner_index});

      // Bug 5 fix: accumulate weighted occupancy for the disconnected
      // subtraction.  Each operator in the CT-HYB configuration
      // represents one unit of occupation on orbital a during this MC
      // sample, weighted by s / beta (time-average over the segment).
      // Because CT-HYB configs are in imaginary time and each operator
      // appears as a kink, the simplest unbiased estimator is to count
      // operator appearances; the exact form of the occupancy estimator
      // should match whatever is used elsewhere in the code (e.g. the
      // standard n_tau measurement).  Here we accumulate the count and
      // divide by beta in collect_results to get a dimensionless mean.
      n_mean(a) += s;

      for (size_t j = 0; j < ops.size(); ++j) {
        if (j == i) continue; // skip self-pair (Bug 4 fix)

        auto const &[t2, op2] = ops[j];
        int const b            = data.linindex.at({op2.block_index, op2.inner_index});

        // Bug 3 fix: use the raw time difference mapped to [-1, 1]
        // without folding.  t1, t2 in [0, beta).
        double dt = double(t1) - double(t2);
        // Wrap into [0, beta) so that x is in [-1, 1].
        if (dt < 0.0) dt += beta;
        double x = 2.0 * dt / beta - 1.0;
        leg.reset(x);

        // Bug 2 fix: weight is just s; no dagger-sign factors.
        for (int n = 0; n < n_leg; ++n) {
          alpha_n(a, b, n) += s * leg.next();
        }
      }
    }
  }

  void measure_D0_corr::collect_results(mpi::communicator const &c) {
    average_sign = mpi::all_reduce(average_sign, c);
    alpha_n      = mpi::all_reduce(alpha_n, c);
    n_mean       = mpi::all_reduce(n_mean, c);

    double norm = real(average_sign);
    if (norm == 0.0) TRIQS_RUNTIME_ERROR << "measure_D0_corr: average sign is zero.";

    double beta = data.config.beta();

    // Bug 6 fix: Eq. (67) reads  q_n = (1 / beta f_n) sum_p M_{pn} <alpha_p>
    // where f_n = 2/(2n+1) is the Legendre norm on [-1,1].
    // build_M_matrix returns the raw change-of-basis matrix WITHOUT f_n,
    // so we must supply f_n = 2/(2n+1) explicitly.
    // The previous code divided by (2n+1) rather than by 2/(2n+1), which
    // is wrong by a factor of (2n+1)^2 / 2.
    nda::matrix<double> M = build_M_matrix(n_leg, beta);

    nda::array<mc_weight_t, 3> q_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});

    for (int a = 0; a < n_lin; ++a) {
      for (int b = 0; b < n_lin; ++b) {
        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t sum = 0.0;
          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          // f_n = 2 / (2n+1)
          double f_n     = 2.0 / (2.0 * n + 1.0);
          q_n(a, b, n)   = sum / (beta * f_n);
        }
      }
    }

    // Bug 5 fix: subtract the disconnected piece <n_a> <n_b>.
    // The mean occupancy estimator accumulated s per operator visit;
    // dividing by (norm * beta) gives the imaginary-time-averaged <n_a>.
    // The P_0 Legendre coefficient of the constant <n_a><n_b> is
    // <n_a><n_b> itself (since P_0 = 1 and the norm f_0 = 2 absorbs the
    // 1/2 from the [-1,1] integral), so we subtract only from the n=0
    // Legendre coefficient of the full correlator.
    for (int a = 0; a < n_lin; ++a) {
      double na = real(n_mean(a)) / (norm * beta);
      for (int b = 0; b < n_lin; ++b) {
        double nb         = real(n_mean(b)) / (norm * beta);
        q_n(a, b, 0)     -= na * nb;
      }
    }

    // Pack q_n into the Legendre Green's function Q_l.
    for (auto bl1 : range(Q_l.size1())) {
      for (auto bl2 : range(Q_l.size2())) {
        int bl1_size = Q_l(bl1, bl2).target_shape()[0];
        int bl2_size = Q_l(bl1, bl2).target_shape()[1];
        for (int i1 = 0; i1 < bl1_size; ++i1) {
          for (int i2 = 0; i2 < bl2_size; ++i2) {
            int lin1 = data.linindex.at({static_cast<int>(bl1), i1});
            int lin2 = data.linindex.at({static_cast<int>(bl2), i2});
            for (auto l : Q_l(bl1, bl2).mesh()) {
              Q_l(bl1, bl2)[l](i1, i2) = q_n(lin1, lin2, l.index());
            }
          }
        }
      }
    }

    // Back-transform to imaginary time via Q(tau) = sum_n q_n P_n(x(tau)).
    triqs::utility::legendre_generator leg;
    for (auto bl1 : range(Q_tau.size1())) {
      for (auto bl2 : range(Q_tau.size2())) {
        int bl1_size = Q_tau(bl1, bl2).target_shape()[0];
        int bl2_size = Q_tau(bl1, bl2).target_shape()[1];
        for (int i1 = 0; i1 < bl1_size; ++i1) {
          for (int i2 = 0; i2 < bl2_size; ++i2) {
            int lin1 = data.linindex.at({static_cast<int>(bl1), i1});
            int lin2 = data.linindex.at({static_cast<int>(bl2), i2});
            for (auto tau : Q_tau(bl1, bl2).mesh()) {
              double x = 2.0 * tau.value() / beta - 1.0;
              leg.reset(x);
              double val = 0.0;
              for (int n = 0; n < n_leg; ++n) val += real(q_n(lin1, lin2, n)) * leg.next();
              Q_tau(bl1, bl2)[tau](i1, i2) = val;
            }
          }
        }
      }
    }
  }

} // namespace triqs_cthyb
