#include <triqs/mc_tools.hpp>
#include <triqs/utility/legendre.hpp>

#include "./D0_corr.hpp"
#include "../math_utils.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_D0_corr::measure_D0_corr(std::optional<Q_l_t> &Q_l_opt, std::optional<Q_tau_t> &Q_tau_opt,
                                   std::optional<Q_conserved_l_t> &Q_conserved_l_opt, std::optional<Q_conserved_tau_t> &Q_conserved_tau_opt,
                                   qmc_data const &data, int n_tau, int n_leg, gf_struct_t const &gf_struct,
                                   std::vector<nda::vector<double>> const &conserved_vectors)
     : data(data), average_sign(0), n_leg(n_leg), conserved_vectors(conserved_vectors) {

    // Only the occupation kinks enter, not the Lang-Firsov kernel. They determine exactly the
    // correlators of the density combinations that commute with h_loc (piecewise constant between
    // trace operators), and nothing else, with or without Lang-Firsov: see
    // doc/notes/dynamical_interactions.tex, "Measuring correlators as coupling derivatives".
    if (n_leg <= 0) TRIQS_RUNTIME_ERROR << "measure_D0_corr requires n_leg > 0, got " << n_leg;

    n_lin = static_cast<int>(data.linindex.size());
    if (n_lin <= 0) TRIQS_RUNTIME_ERROR << "measure_D0_corr requires non-empty linindex.";

    n_conserved = static_cast<int>(conserved_vectors.size());
    if (n_conserved == 0)
      TRIQS_RUNTIME_ERROR << "measure_D0_corr: no combination of orbital densities commutes with h_loc, so the kink estimator determines nothing.";

    // Every n_a commutes with h_loc: the conserved basis is the n_a themselves
    orbital_resolved = (n_conserved == n_lin);

    double beta         = data.config.beta();
    Q_conserved_l_opt   = Q_conserved_l_t({beta, Boson, n_leg}, {n_conserved, n_conserved});
    Q_conserved_tau_opt = Q_conserved_tau_t({beta, Boson, n_tau}, {n_conserved, n_conserved});
    Q_conserved_l.rebind(*Q_conserved_l_opt);
    Q_conserved_tau.rebind(*Q_conserved_tau_opt);
    Q_conserved_l()   = 0.0;
    Q_conserved_tau() = 0.0;

    if (orbital_resolved) {
      Q_l_opt   = make_block2_gf<legendre>({beta, Boson, n_leg}, gf_struct);
      Q_tau_opt = make_block2_gf<imtime>({beta, Boson, n_tau}, gf_struct);

      Q_l.rebind(*Q_l_opt);
      Q_tau.rebind(*Q_tau_opt);

      Q_l()   = 0.0;
      Q_tau() = 0.0;
    } else {
      Q_l_opt.reset();
      Q_tau_opt.reset();
    }

    alpha_n = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});
  }

  void measure_D0_corr::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    // Every occupation kink in the trace, including the stochastic dynamical vertices' operators
    auto ops = data.trace_ops();

    if (ops.size() < 2) return;

    double beta = data.config.beta();

    // Each unordered pair once: the (j, i) term is the (i, j) term at beta - dt, i.e. at -x,
    // and P_n(-x) = (-1)^n P_n(x), so one Legendre recursion serves both orderings.
    for (size_t i = 0; i < ops.size(); ++i) {
      for (size_t j = i + 1; j < ops.size(); ++j) {
        long const a      = ops[i].second.linear_index;
        long const b      = ops[j].second.linear_index;
        double const s1s2 = (ops[i].second.dagger == ops[j].second.dagger) ? 1.0 : -1.0;

        double dt = double(ops[i].first - ops[j].first);
        // The two operators of one stochastic dynamical vertex sit one tick (tau_seg epsilon) apart:
        // a single event, i.e. a contact term at tau = 0 like i == j, not two kinks at separation dt.
        if (dt < 1e-10 * beta || dt > beta * (1.0 - 1e-10)) continue;

        triqs::utility::legendre_generator leg;
        leg.reset(2.0 * dt / beta - 1.0);

        mc_weight_t const w = s * s1s2;
        for (int n = 0; n < n_leg; ++n) {
          double const p = leg.next();
          alpha_n(a, b, n) += w * p;
          alpha_n(b, a, n) += (n % 2 == 0 ? w : -w) * p;
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

    // Enforce KMS symmetry: Q_ab(tau) = Q_ba(beta - tau) => q_n(a, b) = (-1)^n q_n(b, a)
    nda::array<mc_weight_t, 3> q_n_symm = nda::zeros<mc_weight_t>(std::array<long, 3>{n_lin, n_lin, n_leg});
    for (int a = 0; a < n_lin; ++a) {
      for (int b = 0; b < n_lin; ++b) {
        for (int n = 0; n < n_leg; ++n) {
          double sign = (n % 2 == 0) ? 1.0 : -1.0;
          q_n_symm(a, b, n) = 0.5 * (q_n(a, b, n) + sign * q_n(b, a, n));
        }
      }
    }
    q_n = q_n_symm;

    // Contract with the conserved combinations: <O_i(tau) O_j(0)> = sum_ab v_i[a] v_j[b] Q_ab(tau)
    for (auto l : Q_conserved_l.mesh()) {
      for (int i = 0; i < n_conserved; ++i) {
        for (int j = 0; j < n_conserved; ++j) {
          mc_weight_t sum = 0.0;
          for (int a = 0; a < n_lin; ++a)
            for (int b = 0; b < n_lin; ++b) sum += conserved_vectors[i](a) * conserved_vectors[j](b) * q_n(a, b, l.index());
          Q_conserved_l[l](i, j) = sum;
        }
      }
    }
    triqs::utility::legendre_generator leg_conserved;
    for (auto tau : Q_conserved_tau.mesh()) {
      leg_conserved.reset(2.0 * tau.value() / beta - 1.0);
      Q_conserved_tau[tau] = 0.0;
      for (auto l : Q_conserved_l.mesh()) Q_conserved_tau[tau] += leg_conserved.next() * Q_conserved_l[l];
    }

    if (!orbital_resolved) return;

    // Pack q_n into the block2 Legendre Green's function and reconstruct Q_tau
    for (auto bl1 : range(Q_l.size1())) {
      for (auto bl2 : range(Q_l.size2())) {
        
        // 1. Pack Q_l
        for (auto l : Q_l(bl1, bl2).mesh()) {
          for (int i1 = 0; i1 < Q_l(bl1, bl2).target_shape()[0]; ++i1) {
            for (int i2 = 0; i2 < Q_l(bl1, bl2).target_shape()[1]; ++i2) {
              int lin1 = data.linindex.at({static_cast<int>(bl1), i1});
              int lin2 = data.linindex.at({static_cast<int>(bl2), i2});
              if (l.index() < n_leg) {
                Q_l(bl1, bl2)[l](i1, i2) = q_n(lin1, lin2, l.index());
              } else {
                Q_l(bl1, bl2)[l](i1, i2) = 0.0;
              }
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
