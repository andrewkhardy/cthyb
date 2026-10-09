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
     : data(data), average_sign(0), n_leg(n_leg), conserved_vectors(conserved_vectors), moments(n_leg) {

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
    pair_x.resize(n_lin * n_lin);
    pair_w.resize(n_lin * n_lin);
  }

  void measure_D0_corr::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    double const beta = data.config.beta();

    // Every occupation kink in the trace, the dynamical vertices' operators included
    kinks.clear();
    data.for_each_trace_op([&](time_pt const &tau, op_desc const &op) { kinks.push_back({double(tau), op.linear_index, op.dagger ? 1.0 : -1.0}); });

    // Each unordered pair once, as the flavours (a, b) of (i, j) at the cyclic dt from j to i. The (b, a) ordering sits at
    // beta - dt, i.e. at -x, where P_n(-x) = (-1)^n P_n(x): collect_results adds it.
    for (auto &x : pair_x) x.clear();
    for (auto &w : pair_w) w.clear();
    for (size_t i = 0; i < kinks.size(); ++i)
      for (size_t j = i + 1; j < kinks.size(); ++j) {
        double dt = kinks[i].tau - kinks[j].tau;
        if (dt < 0) dt += beta;
        // The two operators of a dynamical vertex bilinear sit one tick apart: one event, a contact term like i == j
        if (dt < 1e-10 * beta || dt > beta * (1.0 - 1e-10)) continue;
        long const ab = kinks[i].a * n_lin + kinks[j].a;
        pair_x[ab].push_back(2.0 * dt / beta - 1.0);
        pair_w[ab].push_back(s * (kinks[i].s * kinks[j].s));
      }
    for (long a = 0; a < n_lin; ++a)
      for (long b = 0; b < n_lin; ++b) moments.add(pair_x[a * n_lin + b], pair_w[a * n_lin + b], &alpha_n(a, b, 0));
  }

  void measure_D0_corr::collect_results(mpi::communicator const &c) {
    average_sign = mpi::all_reduce(average_sign, c);
    alpha_n      = mpi::all_reduce(alpha_n, c);

    // Both orderings of every pair: (a, b) at x and (b, a) at -x
    auto const one_ordering = alpha_n;
    for (int a = 0; a < n_lin; ++a)
      for (int b = 0; b < n_lin; ++b)
        for (int n = 0; n < n_leg; ++n) alpha_n(a, b, n) = one_ordering(a, b, n) + (n % 2 == 0 ? 1.0 : -1.0) * one_ordering(b, a, n);

    double const norm = real(average_sign);
    if (norm == 0.0) TRIQS_RUNTIME_ERROR << "measure_D0_corr: average sign is zero.";
    double const beta     = data.config.beta();
    nda::matrix<double> M = build_M_matrix(n_leg, beta);

    // alpha_n(b, a, n) = (-1)^n alpha_n(a, b, n) and M keeps the parity of n, so q_n obeys KMS: q_n(a, b) = (-1)^n q_n(b, a)
    nda::array<mc_weight_t, 3> q_n(n_lin, n_lin, n_leg);
    for (int a = 0; a < n_lin; ++a)
      for (int b = 0; b < n_lin; ++b)
        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t sum = 0.0;
          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          q_n(a, b, n) = -sum * (2.0 * n + 1.0) / (beta * beta);
        }

    // Q(tau) = sum_l P_l(2 tau / beta - 1) Q_l
    auto legendre_to_tau = [beta](auto const &g_l, auto &&g_tau) {
      triqs::utility::legendre_generator leg;
      for (auto tau : g_tau.mesh()) {
        leg.reset(2.0 * tau.value() / beta - 1.0);
        g_tau[tau] = 0.0;
        for (auto l : g_l.mesh()) g_tau[tau] += leg.next() * g_l[l];
      }
    };

    // Contract with the conserved combinations: <O_i(tau) O_j(0)> = sum_ab v_i[a] v_j[b] Q_ab(tau)
    for (auto l : Q_conserved_l.mesh())
      for (int i = 0; i < n_conserved; ++i)
        for (int j = 0; j < n_conserved; ++j) {
          mc_weight_t sum = 0.0;
          for (int a = 0; a < n_lin; ++a)
            for (int b = 0; b < n_lin; ++b) sum += conserved_vectors[i](a) * conserved_vectors[j](b) * q_n(a, b, l.index());
          Q_conserved_l[l](i, j) = sum;
        }
    legendre_to_tau(Q_conserved_l, Q_conserved_tau);

    if (!orbital_resolved) return;

    for (auto bl1 : range(Q_l.size1()))
      for (auto bl2 : range(Q_l.size2())) {
        auto &&Q = Q_l(bl1, bl2);
        for (auto l : Q.mesh())
          for (int i1 = 0; i1 < Q.target_shape()[0]; ++i1)
            for (int i2 = 0; i2 < Q.target_shape()[1]; ++i2)
              Q[l](i1, i2) = q_n(data.linindex.at({int(bl1), i1}), data.linindex.at({int(bl2), i2}), l.index());
        legendre_to_tau(Q, Q_tau(bl1, bl2));
      }
  }

} // namespace triqs_cthyb
