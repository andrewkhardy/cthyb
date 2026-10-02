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

  // <n_a(tau) n_b(0)> for every pair of spin-orbitals, whether or not the n_a commute with h_loc
  class measure_nn_tau {
    public:
    measure_nn_tau(std::optional<Q_tau_t> &nn_tau_opt, qmc_data const &data, int n_tau, gf_struct_t const &gf_struct);
    void accumulate(mc_weight_t s) { sweep.accumulate(s); }
    void collect_results(mpi::communicator const &c);

    private:
    occupation_sweep sweep;
    std::vector<long> offset; // index of the first spin-orbital of each block in the sweep's operators
    Q_tau_t::view_type nn_tau;
  };

} // namespace triqs_cthyb
