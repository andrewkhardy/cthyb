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
    measure_D0_corr(std::optional<Q_l_t> &Q_l_opt, std::optional<Q_tau_t> &Q_tau_opt, qmc_data const &data, int n_tau, int n_leg,
                    gf_struct_t const &gf_struct);

    void accumulate(mc_weight_t s);
    void collect_results(mpi::communicator const &c);

    private:
    qmc_data const &data;
    mc_weight_t average_sign;
    Q_l_t::view_type Q_l;
    Q_tau_t::view_type Q_tau;
    nda::array<mc_weight_t, 3> alpha_n;
    int n_leg;
    int n_lin;
  };

} // namespace triqs_cthyb
