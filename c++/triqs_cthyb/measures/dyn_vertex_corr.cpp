#include <triqs/mc_tools.hpp>
#include <triqs/utility/legendre.hpp>

#include "./dyn_vertex_corr.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_dyn_vertex_corr::measure_dyn_vertex_corr(std::optional<dyn_vertex_corr_tau_t> &corr_tau_opt,
                                                   std::optional<dyn_vertex_hist_l_t> &hist_l_opt, qmc_data const &data, int n_tau, int n_leg)
     : data(data), average_sign(0), n_leg(n_leg) {

    if (n_leg <= 0) TRIQS_RUNTIME_ERROR << "measure_dyn_vertex_corr requires n_leg > 0, got " << n_leg;
    n_types = static_cast<int>(data.dyn_op_list.size());
    if (n_types == 0) TRIQS_RUNTIME_ERROR << "measure_dyn_vertex_corr: there are no stochastic dynamical vertices to measure.";

    // accumulate() indexes by the vertex's own f_index, which fold_into_stochastic_catalog assigns
    // in catalog order
    for (int t = 0; t < n_types; ++t)
      if (data.dyn_op_list[t].f_index != t)
        TRIQS_RUNTIME_ERROR << "measure_dyn_vertex_corr: dynamical vertex catalog entry " << t << " has coupling index "
                            << data.dyn_op_list[t].f_index << ", expected one coupling per entry in order.";

    double const beta = data.config.beta();
    corr_tau_opt      = dyn_vertex_corr_tau_t(n_types, gf<imtime, scalar_valued>{{beta, Boson, n_tau}});
    hist_l_opt        = dyn_vertex_hist_l_t(n_types, gf<legendre, scalar_valued>{{beta, Boson, n_leg}});
    corr_tau          = &(*corr_tau_opt);
    hist_l            = &(*hist_l_opt);
    for (auto &g : *corr_tau) g() = 0.0;
    for (auto &g : *hist_l) g() = 0.0;

    // The reverse of each catalog entry, for the fold in collect_results
    reversed_type.assign(n_types, -1);
    for (int t = 0; t < n_types; ++t)
      for (int u = 0; u < n_types; ++u)
        if (data.dyn_op_list[u].op1 == data.dyn_op_list[t].op2 && data.dyn_op_list[u].op2 == data.dyn_op_list[t].op1) reversed_type[t] = u;

    hist_n = nda::zeros<mc_weight_t>(std::array<long, 2>{n_types, n_leg});
  }

  // --------------------

  void measure_dyn_vertex_corr::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    double const beta = data.config.beta();
    triqs::utility::legendre_generator leg;
    for (auto const &vertex : data.config.dyn_oplist) {
      // insert_dyn and swap_dyn both keep op1 at the later time, so this is in (0, beta)
      double const separation = double(vertex.tau1 - vertex.tau2);
      leg.reset(2.0 * separation / beta - 1.0);
      for (int l = 0; l < n_leg; ++l) hist_n(vertex.ops.f_index, l) += s * leg.next();
    }
  }

  // --------------------

  void measure_dyn_vertex_corr::collect_results(mpi::communicator const &c) {
    average_sign = mpi::all_reduce(average_sign, c);
    hist_n       = mpi::all_reduce(hist_n, c);

    double const norm = real(average_sign);
    if (norm == 0.0) TRIQS_RUNTIME_ERROR << "measure_dyn_vertex_corr: average sign is zero.";
    double const beta = data.config.beta();

    for (int t = 0; t < n_types; ++t)
      for (auto l : (*hist_l)[t].mesh()) (*hist_l)[t][l] = hist_n(t, l.index()) / norm;

    // H_t(s) = sum_l (2l+1)/beta * <sum_v P_l(x(s_v))> * P_l(x(s))
    auto histogram_at = [&](int t, double s) {
      triqs::utility::legendre_generator leg;
      leg.reset(2.0 * s / beta - 1.0);
      double value = 0.0;
      for (int l = 0; l < n_leg; ++l) value += (2.0 * l + 1.0) / beta * real(hist_n(t, l)) / norm * leg.next();
      return value;
    };

    for (int t = 0; t < n_types; ++t) {
      int const reverse = reversed_type[t];
      auto const &f     = data.dyn_interactions[t];
      long const n_pts  = (*corr_tau)[t].mesh().size();
      std::vector<double> numerator(n_pts), denominator(n_pts);
      double largest_denominator = 0.0;

      for (auto const &tau : (*corr_tau)[t].mesh()) {
        double const s = tau.value();
        double num     = histogram_at(t, s);
        double den     = (beta - s) * f(s);
        if (reverse >= 0) {
          num += histogram_at(reverse, beta - s);
          den += s * data.dyn_interactions[reverse](beta - s);
        }
        numerator[tau.index()]   = num;
        denominator[tau.index()] = den;
        largest_denominator      = std::max(largest_denominator, std::abs(den));
      }

      // Dividing by a vanishing coupling would only amplify noise: leave those points at zero
      for (auto const &tau : (*corr_tau)[t].mesh()) {
        long const i           = tau.index();
        bool const usable      = std::abs(denominator[i]) > 1.e-8 * largest_denominator;
        (*corr_tau)[t][tau]    = usable ? -numerator[i] / denominator[i] : 0.0;
      }
    }
  }

} // namespace triqs_cthyb
