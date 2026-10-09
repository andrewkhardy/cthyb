#include <triqs/operators/many_body_operator.hpp>

#include "./nn_tau.hpp"

namespace triqs_cthyb {

  namespace {
    std::vector<many_body_op_t> densities(gf_struct_t const &gf_struct) {
      std::vector<many_body_op_t> ops;
      for (auto const &[bl, bl_size] : gf_struct)
        for (long i = 0; i < bl_size; ++i) ops.push_back(triqs::operators::n(bl, i));
      return ops;
    }
  } // namespace

  measure_nn_tau::measure_nn_tau(std::optional<Q_tau_t> &nn_tau_opt, std::optional<Q_dlr_t> &nn_dlr_opt, qmc_data const &data,
                                 dlr_imtime const &nodes, int n_tau, gf_struct_t const &gf_struct)
     : sweep(data, nodes, densities(gf_struct), densities(gf_struct)), nodes(nodes), gf_struct(gf_struct) {
    nn_tau_opt = make_block2_gf<imtime>({data.config.beta(), Boson, n_tau}, gf_struct);
    nn_dlr_opt = make_block2_gf<dlr>(dlr{nodes}, gf_struct);
    nn_tau.rebind(*nn_tau_opt);
    nn_dlr.rebind(*nn_dlr_opt);
    nn_tau() = 0.0;
    nn_dlr() = 0.0;
    long n   = 0;
    for (auto const &[bl, bl_size] : gf_struct) {
      offset.push_back(n);
      n += bl_size;
    }
  }

  void measure_nn_tau::collect_results(mpi::communicator const &c) {
    auto const corr = sweep.collect(c);

    // With n_a and n_b swapped at beta - tau, the same correlator: <n_a(tau) n_b(0)> = <n_b(beta - tau) n_a(0)>
    auto at_nodes = make_block2_gf<dlr_imtime>(nodes, gf_struct);
    for (long bl1 = 0; bl1 < at_nodes.size1(); ++bl1)
      for (long bl2 = 0; bl2 < at_nodes.size2(); ++bl2) {
        auto &g = at_nodes(bl1, bl2);
        for (auto const &tau : g.mesh()) {
          long const l = tau.index();
          for (long i1 = 0; i1 < g.target_shape()[0]; ++i1)
            for (long i2 = 0; i2 < g.target_shape()[1]; ++i2) {
              long const a = offset[bl1] + i1, b = offset[bl2] + i2;
              g[tau](i1, i2) = 0.5 * (corr(l, a, b) + corr(sweep.reflected(l), b, a));
            }
        }
      }

    nn_dlr = make_gf_dlr(at_nodes);
    for (long bl1 = 0; bl1 < nn_tau.size1(); ++bl1)
      for (long bl2 = 0; bl2 < nn_tau.size2(); ++bl2)
        for (auto const &tau : nn_tau(bl1, bl2).mesh()) nn_tau(bl1, bl2)[tau] = nn_dlr(bl1, bl2)(tau.value());
  }

} // namespace triqs_cthyb
