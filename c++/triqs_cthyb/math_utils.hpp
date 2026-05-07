#pragma once
#include <nda/nda.hpp>
#include <nda/linalg.hpp>
#include <triqs/utility/legendre.hpp>
#include <vector>
namespace triqs_cthyb {

inline nda::matrix<double> build_M_matrix(int N, double beta) {
    nda::matrix<double> M = nda::zeros<double>(N, N);
    double beta_sq = beta * beta;

    if (N > 0) M(0, 0) = -beta_sq / 12.0;
    if (N > 1) M(1, 1) = -beta_sq / 60.0;

    for (int q = 0; q < N; ++q) {
        if (q >= 2) {
            M(q, q) = -beta_sq / (2.0 * (2*q - 1) * (2*q + 3));
        }
        if (q + 2 < N) {
            M(q + 2, q) = beta_sq / (4.0 * (2*q + 1) * (2*q + 3));
        }
        if (q - 2 >= 0) {
            M(q - 2, q) = beta_sq / (4.0 * (2*q - 1) * (2*q + 1));
        }
    }

    return M;
}

// Project a tau-sampled bosonic function onto the Legendre basis via trapezoidal quadrature.
// Returns the N Legendre coefficients d_n with the standard bosonic normalisation
// d_n = (2n+1)/beta * integral_0^beta D(tau) P_n(2tau/beta - 1) dtau.
inline nda::vector<double> fit_legendre_coeffs(
    int n_pt, double beta, std::function<double(double)> D0_eval,
    int N)
{
    nda::vector<double> d_n = nda::zeros<double>(N);
    double dtau = beta / (n_pt - 1.0);

    for (int i = 0; i < n_pt; ++i) {
        double tau = i * dtau;
        double x = 2.0 * tau / beta - 1.0;
        triqs::utility::legendre_generator gen;
        gen.reset(x);
        for (int n = 0; n < N; ++n) {
            double P_n_x = gen.next();
            double weight = (i == 0 || i == n_pt - 1) ? 0.5 : 1.0;
            d_n(n) += weight * D0_eval(tau) * P_n_x * dtau;
        }
    }
    for (int n = 0; n < N; ++n) {
        d_n(n) *= (2.0 * n + 1.0) / beta;
    }
    return d_n;
}

} // namespace triqs_cthyb
