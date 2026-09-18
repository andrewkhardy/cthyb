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

// Well-resolved regime: w0 beta / 2 = 5, so 60 Legendre coefficients resolve D(tau)
// comfortably. Measured 3.0e-5 relative; the bound keeps ~3x margin.
TEST(ReconstructK, MatchesAnalyticKForEinsteinBoson) {
  auto err = reconstruct(/*beta*/ 10.0, /*w0*/ 1.0, /*lam_sq*/ 2.0, /*n_leg*/ 60, /*n_pt*/ 2001, /*n_tau*/ 500);
  EXPECT_LT(err.max_rel(), 1e-4);
}

// Fewer coefficients must not make it worse here: at w0 beta / 2 = 5 the Legendre series has
// already converged by n_leg = 30, so this pins convergence rather than luck.
TEST(ReconstructK, ConvergedByThirtyCoefficients) {
  auto err = reconstruct(/*beta*/ 10.0, /*w0*/ 1.0, /*lam_sq*/ 2.0, /*n_leg*/ 30, /*n_pt*/ 2001, /*n_tau*/ 500);
  EXPECT_LT(err.max_rel(), 1e-4);
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

// Truncation degrades once the kernel is sharply peaked: at w0 beta / 2 = 75 (the parameters
// the original scratch harness used) 60 coefficients give ~7e-3 relative, two orders worse
// than the resolved case. Documented as a bound, not a target -- if a change makes this
// markedly worse, the Legendre projection has regressed; if it improves it, tighten the bound.
TEST(ReconstructK, TruncationErrorGrowsForPeakedKernel) {
  auto err = reconstruct(/*beta*/ 100.0, /*w0*/ 1.5, /*lam_sq*/ 2.0, /*n_leg*/ 60, /*n_pt*/ 2001, /*n_tau*/ 500);
  EXPECT_LT(err.max_rel(), 2e-2);
  EXPECT_GT(err.max_rel(), 1e-3);
}

MAKE_MAIN;
