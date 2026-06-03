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

    if (ops.size() < 2) return;

    double beta = data.config.beta();

    for (size_t i = 0; i < ops.size(); ++i) {
      for (size_t j = 0; j < ops.size(); ++j) {
        int const a = data.linindex.at({ops[i].second.block_index, ops[i].second.inner_index});
        int const b = data.linindex.at({ops[j].second.block_index, ops[j].second.inner_index});
        double const s1 = ops[i].second.dagger ? +1.0 : -1.0;
        double const s2 = ops[j].second.dagger ? +1.0 : -1.0;

        double dt = double(ops[i].first - ops[j].first);
        if (dt < 0.0) dt += beta;
        if (dt > beta) dt -= beta;

        triqs::utility::legendre_generator leg;
        leg.reset(2.0 * dt / beta - 1.0);

        for (int n = 0; n < n_leg; ++n) {
          alpha_n(a, b, n) += s * s1 * s2 * leg.next();
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

    nda::matrix<double> M = build_M_matrix(n_leg, beta);

    nda::array<mc_weight_t, 3> q_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});

    for (int a = 0; a < n_lin; ++a) {
      for (int b = 0; b < n_lin; ++b) {
        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t sum = 0.0;
          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          q_n(a, b, n) = -sum * (2.0 * n + 1.0) / (beta * beta);
        }
      }
    }

    // Pack q_n into the block2 Legendre Green's function and reconstruct Q_tau
    for (auto bl1 : range(Q_l.size1())) {
      for (auto bl2 : range(Q_l.size2())) {
        
        // 1. Pack Q_l
        for (auto l : Q_l(bl1, bl2).mesh()) {
          for (int i1 = 0; i1 < Q_l(bl1, bl2).target_shape()[0]; ++i1) {
            for (int i2 = 0; i2 < Q_l(bl1, bl2).target_shape()[1]; ++i2) {
              int lin1 = data.linindex.at({static_cast<int>(bl1), i1});
              int lin2 = data.linindex.at({static_cast<int>(bl2), i2});
              Q_l(bl1, bl2)[l](i1, i2) = q_n(lin1, lin2, l.index());
            }
          }
        }

        // 2. Reconstruct Q_tau
        triqs::utility::legendre_generator leg;
        for (auto tau : Q_tau(bl1, bl2).mesh()) {
          double x = 2.0 * tau.value() / beta - 1.0;
          leg.reset(x);
          Q_tau(bl1, bl2)[tau]() = 0.0;
          for (auto l : Q_l(bl1, bl2).mesh()) {
            Q_tau(bl1, bl2)[tau]() += leg.next() * Q_l(bl1, bl2)[l]();
          }
        }
      }
    }
  }

} // namespace triqs_cthyb
