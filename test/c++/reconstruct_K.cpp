// Unit test for the Legendre machinery in math_utils.hpp: fit_legendre_coeffs projects a
// retarded coupling D(tau) onto the Legendre basis, and build_M_matrix maps those
// coefficients to the ones of its double antiderivative K(tau). build_K_n relies on this
// pair, and so do the K'(0) static-shift helpers in
// python/triqs_cthyb/dynamical_interactions.py, so a regression here moves every
// Lang-Firsov result.
//
// For a single Einstein boson the pair is known analytically:
//   D(tau) = lam^2 cosh(w0 (tau - beta/2)) / sinh(w0 beta / 2)
//   K(tau) = (lam^2 / w0^2) [ cosh(w0 (tau - beta/2)) / sinh(w0 beta / 2) - coth(w0 beta / 2) ]
// K(0) = K(beta) = 0 by construction, which is the boundary condition fixing the two
// integration constants.
//
// Was benchmark scratch (c++/triqs_cthyb/tests/reconstruct_K_test.cpp) that printed a max
// error and returned 0 -- and, having an int main() inside the library's
// file(GLOB_RECURSE *.cpp), was compiled into libtriqs_cthyb_c itself.

#include <triqs/test_tools/gfs.hpp>
#include <triqs_cthyb/math_utils.hpp>

#include <functional>
#include <vector>

using namespace triqs_cthyb;
using triqs::utility::legendre_generator;

namespace {

  double analytic_D(double tau, double beta, double w0, double lam_sq) {
    return lam_sq * std::cosh(w0 * (tau - beta / 2.0)) / std::sinh(w0 * beta / 2.0);
  }

  double analytic_K(double tau, double beta, double w0, double lam_sq) {
    return (lam_sq / (w0 * w0)) * (std::cosh(w0 * (tau - beta / 2.0)) / std::sinh(w0 * beta / 2.0) - 1.0 / std::tanh(w0 * beta / 2.0));
  }

  // Largest |K_reconstructed - K_exact| over a uniform tau mesh, and max |K_exact| to
  // normalise it by.
  struct recon_error {
    double max_abs;
    double scale;
    double max_rel() const { return max_abs / scale; }
  };

  recon_error reconstruct(double beta, double w0, double lam_sq, int n_leg, int n_pt, int n_tau) {
    std::function<double(double)> D0_eval = [&](double tau) { return analytic_D(tau, beta, w0, lam_sq); };

    nda::vector<double> d_n = fit_legendre_coeffs(n_pt, beta, D0_eval, n_leg);
    nda::matrix<double> M   = build_M_matrix(n_leg, beta);
    nda::vector<double> k_n = M * d_n;

    recon_error err{0.0, 0.0};
    for (int i = 0; i < n_tau; ++i) {
      double tau = (double(i) / double(n_tau - 1)) * beta;

      legendre_generator gen;
      gen.reset(2.0 * tau / beta - 1.0);
      double k_recon = 0.0;
      for (int n = 0; n < n_leg; ++n) k_recon += k_n(n) * gen.next();

      double k_exact = analytic_K(tau, beta, w0, lam_sq);
      err.max_abs    = std::max(err.max_abs, std::abs(k_recon - k_exact));
      err.scale      = std::max(err.scale, std::abs(k_exact));
    }
    return err;
  }

} // namespace

// The only error left after projection is that of the piecewise-linear interpolant of D between
// the n_pt samples: (w0 dtau)^2 / 12 relative in K, whatever n_leg is.
namespace {
  double interpolation_floor(double beta, double w0, int n_pt) {
    double dtau = beta / (n_pt - 1);
    return (w0 * dtau) * (w0 * dtau) / 12.0;
  }
} // namespace

// Well-resolved regime: w0 beta / 2 = 5, so 60 Legendre coefficients resolve D(tau)
// comfortably and the error is the interpolation floor, 2.08e-6 relative at n_pt = 2001.
TEST(ReconstructK, MatchesAnalyticKForEinsteinBoson) {
  auto err = reconstruct(/*beta*/ 10.0, /*w0*/ 1.0, /*lam_sq*/ 2.0, /*n_leg*/ 60, /*n_pt*/ 2001, /*n_tau*/ 500);
  EXPECT_LT(err.max_rel(), 1e-5);
}

// Fewer coefficients must not make it worse here: at w0 beta / 2 = 5 the Legendre series has
// already converged by n_leg = 30, so this pins convergence rather than luck.
TEST(ReconstructK, ConvergedByThirtyCoefficients) {
  auto err = reconstruct(/*beta*/ 10.0, /*w0*/ 1.0, /*lam_sq*/ 2.0, /*n_leg*/ 30, /*n_pt*/ 2001, /*n_tau*/ 500);
  EXPECT_LT(err.max_rel(), 1e-5);
}

// K(0) = K(beta) = 0 is the boundary condition that fixes build_M_matrix's integration
// constants, so check it directly rather than only through the max-error norm.
TEST(ReconstructK, VanishesAtBothEndpoints) {
  double beta = 10.0, w0 = 1.0, lam_sq = 2.0;
  int n_leg = 60;

  std::function<double(double)> D0_eval = [&](double tau) { return analytic_D(tau, beta, w0, lam_sq); };
  nda::vector<double> d_n = fit_legendre_coeffs(2001, beta, D0_eval, n_leg);
  nda::matrix<double> M   = build_M_matrix(n_leg, beta);
  nda::vector<double> k_n = M * d_n;

  for (double tau : {0.0, beta}) {
    legendre_generator gen;
    gen.reset(2.0 * tau / beta - 1.0);
    double k_recon = 0.0;
    for (int n = 0; n < n_leg; ++n) k_recon += k_n(n) * gen.next();
    EXPECT_NEAR(k_recon, 0.0, 1e-4);
  }
}

// A sharply peaked kernel, w0 beta / 2 = 75: its Legendre coefficients fall as exp(-n^2 / 150),
// so truncation is negligible from n_leg ~ 60 on and the error is the interpolation floor,
// 4.69e-4 relative at n_pt = 2001 -- for every n_leg. It falls by four when n_pt doubles.
// The trapezoidal projection used before aliased here instead: 7.0e-3 at n_leg = 60, 1.2e-2 at
// 150, then non-monotonic in n_leg. (This test used to assert that 7e-3 as truncation error.)
TEST(ReconstructK, PeakedKernelLimitedByTheGridNotByNLeg) {
  double beta = 100.0, w0 = 1.5, lam_sq = 2.0;
  for (int n_pt : {2001, 4001}) {
    double floor = interpolation_floor(beta, w0, n_pt);
    for (int n_leg : {60, 200, 800, 1600}) {
      auto err = reconstruct(beta, w0, lam_sq, n_leg, n_pt, /*n_tau*/ 500);
      EXPECT_NEAR(err.max_rel(), floor, 0.01 * floor) << "n_pt = " << n_pt << ", n_leg = " << n_leg;
    }
  }
}

// A coupling linear in tau is its own interpolant, so d_0 and d_1 are its two coefficients and
// every other d_n vanishes to rounding, however many are asked for. The trapezoidal projection
// gave |d_n| up to 2.8 here at n_leg = 2000, and d_1 off by 1.5e-7.
TEST(ReconstructK, LinearCouplingProjectsExactly) {
  double beta = 10.0, a = 0.7, b = 0.3;
  int n_leg = 2000;
  std::function<double(double)> D0_eval = [&](double tau) { return a + b * (2.0 * tau / beta - 1.0); };
  nda::vector<double> d_n = fit_legendre_coeffs(2001, beta, D0_eval, n_leg);
  EXPECT_NEAR(d_n(0), a, 1e-12);
  EXPECT_NEAR(d_n(1), b, 1e-11);
  double worst = 0.0;
  for (int n = 2; n < n_leg; ++n) worst = std::max(worst, std::abs(d_n(n)));
  EXPECT_LT(worst, 1e-10);
}

MAKE_MAIN;
