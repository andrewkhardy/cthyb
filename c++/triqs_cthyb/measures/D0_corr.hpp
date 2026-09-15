#pragma once

#include <optional>
#include <nda/nda.hpp>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

#include "../qmc_data.hpp"
#include "../types.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  class measure_D0_corr {
    public:
    /// conserved_vectors: the density combinations that commute with h_loc (conserved_densities),
    /// indexed by linear index. Q_conserved is always measured; the orbital-resolved Q_l/Q_tau only
    /// if every n_a commutes with h_loc, and are reset otherwise.
    measure_D0_corr(std::optional<Q_l_t> &Q_l_opt, std::optional<Q_tau_t> &Q_tau_opt, std::optional<Q_conserved_l_t> &Q_conserved_l_opt,
                    std::optional<Q_conserved_tau_t> &Q_conserved_tau_opt, qmc_data const &data, int n_tau, int n_leg,
                    gf_struct_t const &gf_struct, std::vector<nda::vector<double>> const &conserved_vectors);

    void accumulate(mc_weight_t s);
    void collect_results(mpi::communicator const &c);

    private:
    qmc_data const &data;
    mc_weight_t average_sign;
    Q_l_t::view_type Q_l;
    Q_tau_t::view_type Q_tau;
    Q_conserved_l_t::view_type Q_conserved_l;
    Q_conserved_tau_t::view_type Q_conserved_tau;
    nda::array<mc_weight_t, 3> alpha_n;
    int n_leg;
    int n_lin;
    std::vector<nda::vector<double>> conserved_vectors;
    int n_conserved;
    bool orbital_resolved;
  };

} // namespace triqs_cthyb
