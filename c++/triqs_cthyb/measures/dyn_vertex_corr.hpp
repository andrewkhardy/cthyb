#pragma once

#include <optional>
#include <vector>
#include <nda/nda.hpp>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

#include "../qmc_data.hpp"
#include "../types.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  /// C_t(s) = <O1_t(s) O2_t(0)> of every stochastic dynamical vertex type t, from the histogram H_t of the vertex
  /// separations s = tau1 - tau2: d ln Z / d f_t gives H_t(s) = -(beta - s) f_t(s) C_t(s), and folding t with its
  /// reverse tbar (op1 <-> op2), C_t(s) = -[H_t(s) + H_tbar(beta - s)] / [(beta - s) f_t(s) + s f_tbar(beta - s)].
  /// Points with a negligible denominator are left at zero; dyn_vertex_hist_l holds the raw histogram.
  class measure_dyn_vertex_corr {
    public:
    measure_dyn_vertex_corr(std::optional<dyn_vertex_corr_tau_t> &corr_tau_opt, std::optional<dyn_vertex_hist_l_t> &hist_l_opt,
                            qmc_data const &data, int n_tau, int n_leg);

    void accumulate(mc_weight_t s);
    void collect_results(mpi::communicator const &c);

    private:
    qmc_data const &data;
    mc_weight_t average_sign;
    int n_leg;
    int n_types;
    nda::array<mc_weight_t, 2> hist_n; // [vertex type, Legendre order]
    std::vector<int> reversed_type;    // catalog entry with op1 and op2 exchanged, -1 if there is none
    dyn_vertex_corr_tau_t *corr_tau;
    dyn_vertex_hist_l_t *hist_l;
  };

} // namespace triqs_cthyb
