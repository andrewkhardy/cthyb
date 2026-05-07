// Simple test: reconstruct K(tau) from D(tau) via Legendre/M matrix
// Compares analytic K(tau) to reconstructed K_recon(tau) and prints max error.

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

#include "math_utils.hpp"

using namespace triqs_cthyb;

double analytic_D(double tau, double beta, double w0, double lam_sq) {
    return lam_sq * std::cosh(w0 * (tau - beta/2.0)) / std::sinh(w0 * beta / 2.0);
}

double analytic_K(double tau, double beta, double w0, double lam_sq) {
    return (lam_sq / (w0*w0)) * (std::cosh(w0 * (tau - beta/2.0)) / std::sinh(w0 * beta / 2.0) - 1.0 / std::tanh(w0 * beta / 2.0));
}

// Evaluate Legendre polynomials up to N-1 at x using recurrence
static void eval_legendre_all(int N, double x, std::vector<double> &out) {
    out.assign(N, 0.0);
    if (N <= 0) return;
    out[0] = 1.0;
    if (N == 1) return;
    out[1] = x;
    for (int n = 1; n + 1 < N; ++n) {
        double Pn = out[n];
        double Pn_1 = out[n-1];
        out[n+1] = ((2.0*n + 1.0) * x * Pn - n * Pn_1) / (n + 1.0);
    }
}

int main() {
    // Parameters (match Python reference)
    double beta = 100.0;
    double w0 = 1.5;
    double lam_sq = 2.0;

    // Legendre resolution
    int N_leg = 60; // try a moderately large number

    // Quadrature / mesh sizes (we follow compute_D_legendre_coeffs sampling)
    int n_pt = 2001; // number of tau points used to compute d_n in the C++ helper
    int n_tau = 500; // reconstruction mesh

    // Build D0_eval function
    std::function<double(double)> D0_eval = [&](double tau) {
        return analytic_D(tau, beta, w0, lam_sq);
    };

    // Compute d_n using the helper (uses the same discrete sampling as library)
    nda::vector<double> d_n = compute_D_legendre_coeffs(n_pt, beta, D0_eval, N_leg);

    // Build M and compute k_n = M * d_n
    nda::matrix<double> M = build_M_matrix(N_leg, beta);
    nda::vector<double> k_n = M * d_n;

    // Prepare tau mesh for reconstruction
    std::vector<double> tau_mesh(n_tau);
    std::vector<double> x_mesh(n_tau);
    for (int i = 0; i < n_tau; ++i) {
        tau_mesh[i] = (double(i) / double(n_tau - 1)) * beta;
        x_mesh[i] = 2.0 * tau_mesh[i] / beta - 1.0;
    }

    // Reconstruct K on tau_mesh and compare to analytic
    double max_error = 0.0;
    std::vector<double> Pvals;
    for (int i = 0; i < n_tau; ++i) {
        eval_legendre_all(N_leg, x_mesh[i], Pvals);
        double K_recon = 0.0;
        for (int n = 0; n < N_leg; ++n) K_recon += k_n(n) * Pvals[n];
        double K_exact = analytic_K(tau_mesh[i], beta, w0, lam_sq);
        double err = std::abs(K_recon - K_exact);
        if (err > max_error) max_error = err;
    }

    std::cout << "Reconstruction test: N_leg=" << N_leg << " n_pt=" << n_pt << " n_tau=" << n_tau << std::endl;
    std::cout << "Max |K_recon - K_exact| = " << max_error << std::endl;

    // Print a few sample points for manual inspection
    for (int j : {0, n_tau/4, n_tau/2, 3*n_tau/4, n_tau-1}) {
        eval_legendre_all(N_leg, x_mesh[j], Pvals);
        double K_recon = 0.0;
        for (int n = 0; n < N_leg; ++n) K_recon += k_n(n) * Pvals[n];
        double K_exact = analytic_K(tau_mesh[j], beta, w0, lam_sq);
        std::cout << "tau=" << tau_mesh[j] << " K_exact=" << K_exact << " K_recon=" << K_recon << " err=" << std::abs(K_recon-K_exact) << std::endl;
    }

    return 0;
}
