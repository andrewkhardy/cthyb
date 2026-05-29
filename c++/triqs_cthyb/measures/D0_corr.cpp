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

    // n_lin = total number of flattened orbital indices from the linindex map,
    // NOT K_n.size() which equals n_leg (one entry per Legendre order, not per orbital).
    n_lin = static_cast<int>(data.linindex.size());
    if (n_lin <= 0) TRIQS_RUNTIME_ERROR << "measure_D0_corr requires non-empty linindex.";

    Q_l_opt   = make_block2_gf<legendre>({data.config.beta(), Boson, n_leg}, gf_struct);
    Q_tau_opt = make_block2_gf<imtime>({data.config.beta(), Boson, n_tau}, gf_struct);

    Q_l.rebind(*Q_l_opt);
    Q_tau.rebind(*Q_tau_opt);

    Q_l()   = 0.0;
    Q_tau() = 0.0;

    alpha_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});
  }

  void measure_D0_corr::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    std::vector<std::pair<time_pt, op_desc>> ops;
    ops.reserve(data.config.size());
    for (auto const &entry : data.config) ops.push_back(entry);

    if (ops.empty()) return;

    double beta = data.config.beta();
    triqs::utility::legendre_generator leg_ij, leg_ji;

    // -----------------------------------------------------------------------
    // Accumulate alpha_n(a,b,n) = sum_{alpha,beta} S_alpha S_beta P_n(tau_alpha - tau_beta)
    // per Eq. (62), including the diagonal alpha=beta self-pairs.
    //
    // S_alpha = +1 always: the Lang-Firsov weight couples to the density n=c†c
    // which is positive definite; the dagger structure of individual CT-HYB
    // operators is not the same thing and must NOT appear here.
    //
    // The full ordered double-sum is computed using the j>i half-loop trick:
    // for each unordered pair {i,j} we evaluate BOTH time orderings
    // P_n(tau_i - tau_j) and P_n(tau_j - tau_i) and add them together.
    // This is exactly equivalent to the full j!=i double loop but avoids
    // any ambiguity about factor-of-2 conventions in build_M_matrix.
    //
    // The diagonal self-pairs alpha=beta contribute P_n(0) = P_n(x=-1) since
    // dt=0 maps to x = 2*0/beta - 1 = -1.  These are physically real
    // (they represent n_a^2 = n_a for fermions) and must NOT be dropped.
    //
    // Time differences are mapped to x in [-1,1] via x = 2*dt/beta - 1
    // where dt in [0,beta).  There is NO folding to [0,beta/2]: the full
    // Legendre basis on [-1,1] requires x to span the whole interval.
    // -----------------------------------------------------------------------

    // Diagonal self-pairs: alpha = beta, dt = 0 => x = -1, P_n(-1) = (-1)^n.
    // All S_alpha^2 = 1, weight = s.
    {
      leg_ij.reset(-1.0);
      // Pre-compute P_n(-1) once; it is the same for every operator.
      std::vector<double> Pn_self(n_leg);
      for (int n = 0; n < n_leg; ++n) Pn_self[n] = leg_ij.next();

      for (size_t i = 0; i < ops.size(); ++i) {
        auto const &[t1, op1] = ops[i];
        int const a            = data.linindex.at({op1.block_index, op1.inner_index});
        for (int n = 0; n < n_leg; ++n) alpha_n(a, a, n) += s * Pn_self[n];
      }
    }

    // Off-diagonal pairs: alpha != beta.  Use j>i half-loop, sum both orderings.
    for (size_t i = 0; i < ops.size(); ++i) {
      auto const &[t1, op1] = ops[i];
      int const a            = data.linindex.at({op1.block_index, op1.inner_index});

      for (size_t j = i + 1; j < ops.size(); ++j) {
        auto const &[t2, op2] = ops[j];
        int const b            = data.linindex.at({op2.block_index, op2.inner_index});

        // dt_ij = tau_i - tau_j wrapped to [0, beta)
        double dt_ij = double(t1) - double(t2);
        if (dt_ij < 0.0) dt_ij += beta;
        // dt_ji = tau_j - tau_i = beta - dt_ij  (also in [0, beta))
        double dt_ji = beta - dt_ij;

        double x_ij = 2.0 * dt_ij / beta - 1.0;
        double x_ji = 2.0 * dt_ji / beta - 1.0;

        leg_ij.reset(x_ij);
        leg_ji.reset(x_ji);

        for (int n = 0; n < n_leg; ++n) {
          // Both time orderings of the pair {i,j}: covers (alpha=i,beta=j)
          // and (alpha=j,beta=i) from the double sum in Eq. (62).
          double contrib = s * (leg_ij.next() + leg_ji.next());
          alpha_n(a, b, n) += contrib;
          if (a != b) alpha_n(b, a, n) += contrib;
          // When a==b the two orbital indices are the same so alpha_n(a,a,n)
          // already collects both; adding contrib once is correct because
          // alpha_n(a,b) and alpha_n(b,a) are the same array element.
        }
      }
    }
  }

  void measure_D0_corr::collect_results(mpi::communicator const &c) {
    average_sign = mpi::all_reduce(average_sign, c);
    alpha_n      = mpi::all_reduce(alpha_n, c);

    double norm = real(average_sign);
    if (norm == 0.0) TRIQS_RUNTIME_ERROR << "measure_D0_corr: average sign is zero.";

    double beta = data.config.beta();

    // Eq. (67):  q_n = (1 / beta f_n) sum_p M_{pn} <alpha_p>
    // where f_n = 2/(2n+1) is the Legendre orthogonality norm on [-1,1].
    // build_M_matrix returns the raw basis-change matrix WITHOUT f_n,
    // so we divide by (beta * f_n) = beta * 2/(2n+1).
    // Equivalently: multiply by (2n+1) / (2*beta).
    nda::matrix<double> M = build_M_matrix(n_leg, beta);

    nda::array<mc_weight_t, 3> q_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});

    for (int a = 0; a < n_lin; ++a) {
      for (int b = 0; b < n_lin; ++b) {
        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t sum = 0.0;
          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          double f_n   = 2.0 / (2.0 * n + 1.0);
          q_n(a, b, n) = sum / (beta * f_n);
        }
      }
    }

    // No disconnected subtraction: Q(tau) = <n(tau)n(0)> is the FULL
    // correlator as defined in Eq. (59)-(67).  The paper differentiates
    // the full free energy F = -ln Z with respect to d_n, which gives the
    // full (not connected) correlator.  The disconnected piece <n>^2 is
    // tau-independent and sits entirely in q_0; it should be left in place
    // so that dF/dd_n is computed correctly.  Subtracting it here would
    // give the connected correlator, which is a different (and for this
    // purpose wrong) quantity.

    // Pack q_n into the block2 Legendre Green's function.
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

    // Back-transform to imaginary time: Q(tau) = sum_n q_n P_n(x(tau)).
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
