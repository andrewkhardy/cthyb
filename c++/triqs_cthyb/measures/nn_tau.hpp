#pragma once

#include <optional>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

#include "../qmc_data.hpp"
#include "../types.hpp"
#include "./occupation_sweep.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  // <n_a(tau) n_b(0)> for every pair of spin-orbitals, whether or not the n_a commute with h_loc: measured at the DLR
  // nodes, kept as DLR coefficients (nn_dlr) and evaluated on the regular mesh of n_tau points (nn_tau)
  class measure_nn_tau {
    public:
    measure_nn_tau(std::optional<Q_tau_t> &nn_tau_opt, std::optional<Q_dlr_t> &nn_dlr_opt, qmc_data const &data, dlr_imtime const &nodes,
                   int n_tau, gf_struct_t const &gf_struct);
    void accumulate(mc_weight_t s) { sweep.accumulate(s); }
    void collect_results(mpi::communicator const &c);

    private:
    occupation_sweep sweep;
    dlr_imtime nodes;
    gf_struct_t gf_struct;
    std::vector<long> offset; // index of the first spin-orbital of each block in the sweep's operators
    Q_tau_t::view_type nn_tau;
    Q_dlr_t::view_type nn_dlr;
  };

} // namespace triqs_cthyb
