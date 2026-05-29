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

    // FIX: n_lin must be the number of flattened orbital indices from linindex,
    // NOT K_n.size() which equals n_leg (one expansion coefficient per Legendre
    // order, not one per orbital).  Using K_n.size() made alpha_n shaped
    // [n_leg x n_leg x n_leg] instead of [n_orb x n_orb x n_leg].
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
    triqs::utility::legendre_generator leg;

    // Accumulate alpha_n(a,b,n) = sum_{alpha,beta} S_alpha S_beta P_n(tau_alpha - tau_beta)
    // per Eq. (62), including the diagonal alpha=beta self-pairs.
    //
    // The folding convention dt -> [0,beta/2] with x -> [-1,0] is kept because
    // build_M_matrix is constructed against this convention (verified numerically).
    // Changing the x range without rebuilding M would destroy the normalization.
    //
    // FIX: S_alpha = +1 for all operators.  The Lang-Firsov weight couples to
    // the occupation number n_a = c†_a c_a which is always non-negative; the
    // sign of an individual CT-HYB operator (creation vs annihilation) is a
    // property of the action expansion, not of the density in the correlator.
    // Using op.dagger ? +1 : -1 as S_alpha was physically wrong: it made the
    // cross-pair weight antisymmetric under exchange of creation/annihilation
    // operators, turning what should be a positive-definite estimator for
    // <n(tau)n(0)> into a signed quantity with no physical meaning.

    // Self-pairs: alpha=beta, dt=0 => folded dt=0 => x=-1.  S_alpha^2=1.
    leg.reset(-1.0);
    std::vector<double> Pn_self(n_leg);
    for (int n = 0; n < n_leg; ++n) Pn_self[n] = leg.next();

    for (size_t i = 0; i < ops.size(); ++i) {
      auto const &[t1, op1] = ops[i];
      int const a            = data.linindex.at({op1.block_index, op1.inner_index});
      for (int n = 0; n < n_leg; ++n) alpha_n(a, a, n) += s * Pn_self[n];
    }

    // Cross-pairs: alpha != beta.  Use j>i half-loop and fold dt into [0,beta/2].
    // Both orderings (i,j) and (j,i) give the same folded |dt| so we account for
    // both by adding val to alpha_n(a,b) AND alpha_n(b,a), which when a==b is the
    // same element giving the factor of 2 needed for the half-loop.
    for (size_t i = 0; i < ops.size(); ++i) {
      auto const &[t1, op1] = ops[i];
      int const a            = data.linindex.at({op1.block_index, op1.inner_index});

      for (size_t j = i + 1; j < ops.size(); ++j) {
        auto const &[t2, op2] = ops[j];
        int const b            = data.linindex.at({op2.block_index, op2.inner_index});

        // Fold dt into [0, beta/2]; both (tau_i - tau_j) and (tau_j - tau_i)
        // give the same |dt| so this correctly counts both orderings.
        double dt = std::abs(double(t1) - double(t2));
        if (dt > beta / 2.0) dt = beta - dt;
        double x = 2.0 * dt / beta - 1.0; // x in [-1, 0]
        leg.reset(x);

        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t val = s * leg.next(); // S_alpha = S_beta = +1
          alpha_n(a, b, n) += val;
          alpha_n(b, a, n) += val; // == alpha_n(a,a,n) += 2*val when a==b
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

    // build_M_matrix entries scale as beta^2 and encode the change-of-basis
    // between the MC-accumulated alpha_n (sums of P_n evaluations under the
    // folded convention) and the Legendre coefficients q_n of Q(tau).
    // The normalization 1/(beta*(2n+1)) is the matching factor for this M;
    // it was verified to be correct by the original code giving the right shape.
    // Changing it to f_n = 2/(2n+1) (appropriate for the unfolded [-1,1] basis)
    // is wrong here because M was built against the folded convention.
    nda::matrix<double> M = build_M_matrix(n_leg, beta);

    nda::array<mc_weight_t, 3> q_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});

    for (int a = 0; a < n_lin; ++a) {
      for (int b = 0; b < n_lin; ++b) {
        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t sum = 0.0;
          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          q_n(a, b, n) = sum / (beta * (2.0 * n + 1.0));
        }
      }
    }

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
