// Simple test: reconstruct K(tau) from D(tau) via Legendre/M matrix
// Compares analytic K(tau) to reconstructed K_recon(tau) and prints max error.
//
// Uses triqs::utility::legendre_generator for all Legendre evaluation —
// no hand-rolled recurrences, no direct <cmath> includes.

#include "../math_utils.hpp"  // fit_legendre_coeffs, build_M_matrix; also pulls in
                               // nda, triqs/utility/legendre.hpp, and std math transitively

#include <iostream>
#include <vector>
#include <functional>

using namespace triqs_cthyb;
using triqs::utility::legendre_generator;

// ---------------------------------------------------------------------------
// Analytic reference for a single Einstein boson:
//   D(tau)  = (lam^2) * cosh(w0*(tau - beta/2)) / sinh(w0*beta/2)
//   K(tau)  = (lam^2/w0^2) * [ cosh(w0*(tau-beta/2))/sinh(w0*beta/2)
//                               - coth(w0*beta/2) ]
// (All transcendental functions are available via nda / triqs transitive headers.)
// ---------------------------------------------------------------------------
double analytic_D(double tau, double beta, double w0, double lam_sq) {
    return lam_sq * std::cosh(w0 * (tau - beta / 2.0)) / std::sinh(w0 * beta / 2.0);
}

double analytic_K(double tau, double beta, double w0, double lam_sq) {
    return (lam_sq / (w0 * w0))
         * (std::cosh(w0 * (tau - beta / 2.0)) / std::sinh(w0 * beta / 2.0)
            - 1.0 / std::tanh(w0 * beta / 2.0));
}

int main() {
    // Parameters (match Python reference)
    double beta   = 100.0;
    double w0     = 1.5;
    double lam_sq = 2.0;

    // Legendre resolution
    int N_leg = 60;

    // Quadrature / mesh sizes (fit_legendre_coeffs uses trapezoidal sampling)
    int n_pt  = 2001; // tau points used to project D(tau) onto Legendre basis
    int n_tau = 500;  // reconstruction mesh

    // Build D0_eval function
    std::function<double(double)> D0_eval = [&](double tau) {
        return analytic_D(tau, beta, w0, lam_sq);
    };

    // Project D(tau) onto Legendre basis using trapezoidal quadrature
    nda::vector<double> d_n = fit_legendre_coeffs(n_pt, beta, D0_eval, N_leg);

    // Build M and compute k_n = M * d_n
    nda::matrix<double>  M   = build_M_matrix(N_leg, beta);
    nda::vector<double> k_n = M * d_n;

    // Prepare tau mesh for reconstruction
    std::vector<double> tau_mesh(n_tau);
    std::vector<double> x_mesh(n_tau);
    for (int i = 0; i < n_tau; ++i) {
        tau_mesh[i] = (double(i) / double(n_tau - 1)) * beta;
        x_mesh[i]   = 2.0 * tau_mesh[i] / beta - 1.0;
    }

    // Reconstruct K on tau_mesh using triqs::utility::legendre_generator
    double max_error = 0.0;
    for (int i = 0; i < n_tau; ++i) {
        legendre_generator gen;
        gen.reset(x_mesh[i]);
        double K_recon = 0.0;
        for (int n = 0; n < N_leg; ++n) K_recon += k_n(n) * gen.next();
        double K_exact = analytic_K(tau_mesh[i], beta, w0, lam_sq);
        double err = K_recon - K_exact;
        if (err < 0.0) err = -err;
        if (err > max_error) max_error = err;
    }

    std::cout << "Reconstruction test: N_leg=" << N_leg
              << " n_pt=" << n_pt << " n_tau=" << n_tau << std::endl;
    std::cout << "Max |K_recon - K_exact| = " << max_error << std::endl;

    // Print a few sample points for manual inspection
    for (int j : {0, n_tau / 4, n_tau / 2, 3 * n_tau / 4, n_tau - 1}) {
        legendre_generator gen;
        gen.reset(x_mesh[j]);
        double K_recon = 0.0;
        for (int n = 0; n < N_leg; ++n) K_recon += k_n(n) * gen.next();
        double K_exact = analytic_K(tau_mesh[j], beta, w0, lam_sq);
        double err = K_recon - K_exact;
        if (err < 0.0) err = -err;
        std::cout << "tau=" << tau_mesh[j]
                  << " K_exact=" << K_exact
                  << " K_recon=" << K_recon
                  << " err=" << err << std::endl;
    }

    return 0;
}
