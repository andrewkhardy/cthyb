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

  /// Correlator of the two bilinears of every stochastic dynamical vertex type, from the histogram of
  /// vertex separations. Differentiating ln Z with respect to that type's coupling f_t gives
  ///   H_t(s) = <sum_{v in t} delta(s - s_v)> = -(beta - s) f_t(s) C_t(s),  C_t(s) = <O1_t(s) O2_t(0)>,
  /// with s = tau1 - tau2 in (0, beta) the vertex's own time separation and (beta - s) the length of the
  /// ordered region at that separation. Folding a type with its reverse t_bar (op1 and op2 exchanged,
  /// C_tbar(beta - s) = C_t(s)) gives what this measure reports,
  ///   C_t(s) = -[H_t(s) + H_tbar(beta - s)] / [(beta - s) f_t(s) + s f_tbar(beta - s)].
  /// See doc/notes/dynamical_interactions.tex, "Measuring correlators as coupling derivatives".
  ///
  /// Exact, and free (only vertex times are read), but it divides by the coupling, so it is noisy
  /// wherever |f_t| is small and says nothing about channels with no vertices at all. Points where the
  /// denominator is negligible are left at zero; dyn_vertex_hist_l holds the raw histogram's Legendre
  /// coefficients, before any folding or division.
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
