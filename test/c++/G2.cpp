/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2017, H. U.R. Strand
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <triqs_cthyb/solver_core.hpp>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/hilbert_space/fundamental_operator_set.hpp>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>
#include <triqs/test_tools/gfs.hpp>

using namespace triqs_cthyb;
using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::n;
using namespace triqs::gfs;
using namespace triqs::mesh;
using triqs::hilbert_space::gf_struct_t;

TEST(CtHyb, G2_measurments) {

  std::cout << "Welcome to the CTHYB solver\n";

  // Initialize mpi
  int rank = mpi::communicator().rank();

  // Parameters
  double beta = 2.0;
  double U    = 0.0;
  double mu   = 2.0;

  double V1       = 2.0;
  double V2       = 5.0;
  double epsilon1 = 0.0;
  double epsilon2 = 4.0;

  // GF structure
  enum spin { up, down };
  gf_struct_t gf_struct{{"up", 1}, {"down", 1}};
  auto n_up   = n("up", 0);
  auto n_down = n("down", 0);

  // define operators
  auto H = U * n_up * n_down;

  // Construct CTQMC solver
  int n_iw  = 1025;
  int n_tau = 2500;
  int n_l   = 10;
  solver_core solver({beta, gf_struct, n_iw, n_tau, n_l});

  // Set G0
  nda::clef::placeholder<0> om_;
  auto g0_iw = gf<imfreq>{{beta, Fermion}, {1, 1}};
  g0_iw(om_) << om_ + mu - V1 * V1 / (om_ - epsilon1) - V2 * V2 / (om_ - epsilon2);
  for (int bl = 0; bl < 2; ++bl) solver.G0_iw()[bl] = triqs::gfs::inverse(g0_iw);

  // Solve parameters
  int n_cycles      = 500;
  auto p            = solve_parameters_t{.h_int = H, .n_cycles = n_cycles};
  p.random_name     = "";
  p.random_seed     = 123 * rank + 567;
  p.max_time        = -1;
  p.length_cycle    = 100;
  p.n_warmup_cycles = 1000;
  p.move_double     = false;

  p.measure_G2_tau   = true;
  p.measure_G2_n_tau = 3;

  p.measure_G2_iw      = true;
  p.measure_G2_iw_nfft = true;

  p.measure_G2_iw_ph      = true;
  p.measure_G2_iw_ph_nfft = true;
  p.measure_G2_iw_pp      = true;
  p.measure_G2_iw_pp_nfft = true;

  p.measure_G2_n_fermionic = 3;
  p.measure_G2_n_bosonic   = 5;

  p.measure_G2_iwll_pp = true;
  p.measure_G2_n_l     = 3;

  p.nfft_buf_sizes = {{"up", 100}, {"down", 100}};

  // Solve!
  solver.solve(p);

  // Compare standard and nfft measures

  EXPECT_GF_NEAR((*solver.G2_iw)(0, 1), (*solver.G2_iw_nfft)(0, 1));
  EXPECT_GF_NEAR((*solver.G2_iw_ph)(0, 1), (*solver.G2_iw_ph_nfft)(0, 1));
  EXPECT_GF_NEAR((*solver.G2_iw_pp)(0, 1), (*solver.G2_iw_pp_nfft)(0, 1));

  std::cout << "--> solver done, now writing and reading the results.\n";

  // Save the results
  std::string filename = "G2";

  if (rank == 0) {
    h5::file G_file(filename + ".out.h5", 'w');
    if (solver.G2_tau) h5_write(G_file, "G2_tau", (*solver.G2_tau)(0, 1));
    if (solver.G2_iw) h5_write(G_file, "G2_iw", (*solver.G2_iw)(0, 1));
    if (solver.G2_iw_ph) h5_write(G_file, "G2_iw_ph", (*solver.G2_iw_ph)(0, 1));
    if (solver.G2_iw_pp) h5_write(G_file, "G2_iw_pp", (*solver.G2_iw_pp)(0, 1));
    if (solver.G2_iwll_pp) h5_write(G_file, "G2_iwll_pp", (*solver.G2_iwll_pp)(0, 1));
  }

  if (rank == 0) {
    h5::file G_file(filename + ".ref.h5", 'r');

    {
      G2_tau_t::g_t G2_tau;
      h5_read(G_file, "G2_tau", G2_tau);
      if (solver.G2_tau) EXPECT_GF_NEAR(G2_tau, (*solver.G2_tau)(0, 1));
    }

    {
      G2_iw_t::g_t G2_iw;
      h5_read(G_file, "G2_iw", G2_iw);
      if (solver.G2_iw) EXPECT_GF_NEAR(G2_iw, (*solver.G2_iw)(0, 1));
    }

    {
      G2_iw_t::g_t G2_iw;
      h5_read(G_file, "G2_iw_ph", G2_iw);
      if (solver.G2_iw_ph) EXPECT_GF_NEAR(G2_iw, (*solver.G2_iw_ph)(0, 1));
    }

    {
      G2_iw_t::g_t G2_iw;
      h5_read(G_file, "G2_iw_pp", G2_iw);
      if (solver.G2_iw_pp) EXPECT_GF_NEAR(G2_iw, (*solver.G2_iw_pp)(0, 1));
    }

    {
      G2_iwll_t::g_t G2_iwll_pp;
      h5_read(G_file, "G2_iwll_pp", G2_iwll_pp);
      if (solver.G2_iwll_pp) EXPECT_GF_NEAR(G2_iwll_pp, (*solver.G2_iwll_pp)(0, 1));
    }
  }
}

// The Legendre coefficient (l1, l2) of G2_iwll must not depend on how many coefficients are measured. Measurements draw
// no random numbers, so with the same seed the two runs below sample the same configurations, and their common
// coefficients agree to the NFFT's accuracy. The second Legendre generator was once stepped on through every l1
// without being reset, which put P_{l1 n_l + l2} where P_{l2} belongs.
TEST(CtHyb, G2_iwll_independent_of_n_l) {
  double beta = 2.0, mu = 2.0, V1 = 2.0, V2 = 5.0, epsilon1 = 0.0, epsilon2 = 4.0;
  gf_struct_t gf_struct{{"up", 1}, {"down", 1}};

  auto measure = [&](int n_l) {
    solver_core solver({beta, gf_struct, 1025, 2500, 10});
    nda::clef::placeholder<0> om_;
    auto g0_iw = gf<imfreq>{{beta, Fermion}, {1, 1}};
    g0_iw(om_) << om_ + mu - V1 * V1 / (om_ - epsilon1) - V2 * V2 / (om_ - epsilon2);
    for (int bl = 0; bl < 2; ++bl) solver.G0_iw()[bl] = triqs::gfs::inverse(g0_iw);

    auto p                   = solve_parameters_t{.h_int = 1.0 * n("up", 0) * n("down", 0), .n_cycles = 200};
    p.random_seed            = 567;
    p.length_cycle           = 100;
    p.n_warmup_cycles        = 200;
    p.move_double            = false;
    p.verbosity              = 0;
    p.measure_G_tau          = false;
    p.measure_G2_iwll_pp     = true;
    p.measure_G2_iwll_ph     = true;
    p.measure_G2_n_bosonic   = 3;
    p.measure_G2_n_l         = n_l;
    p.nfft_buf_sizes         = {{"up", 100}, {"down", 100}};
    solver.solve(p);
    return std::pair{*solver.G2_iwll_pp, *solver.G2_iwll_ph};
  };

  auto [pp_2, ph_2] = measure(2);
  auto [pp_4, ph_4] = measure(4);

  auto _ = nda::range::all;
  for (int b1 = 0; b1 < 2; ++b1)
    for (int b2 = 0; b2 < 2; ++b2) {
      auto pp_common = pp_4(b1, b2).data()(_, nda::range(2), nda::range(2), _, _, _, _);
      auto ph_common = ph_4(b1, b2).data()(_, nda::range(2), nda::range(2), _, _, _, _);
      EXPECT_ARRAY_NEAR(pp_2(b1, b2).data(), pp_common, 1e-8);
      EXPECT_ARRAY_NEAR(ph_2(b1, b2).data(), ph_common, 1e-8);
    }
}
MAKE_MAIN;
