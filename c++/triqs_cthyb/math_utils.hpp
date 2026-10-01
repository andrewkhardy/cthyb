#pragma once
#include <nda/nda.hpp>
#include <nda/linalg.hpp>
#include <triqs/utility/legendre.hpp>
#include <functional>
#include <vector>

namespace triqs_cthyb {

  // Maps the bosonic Legendre coefficients of D(tau) to those of K(tau), with K'' = D and K(0) = K(beta) = 0
  inline nda::matrix<double> build_M_matrix(int N, double beta) {
    nda::matrix<double> M = nda::zeros<double>(N, N);
    double beta_sq        = beta * beta;

    if (N > 0) M(0, 0) = -beta_sq / 12.0;
    if (N > 1) M(1, 1) = -beta_sq / 60.0;

    for (int q = 0; q < N; ++q) {
      if (q >= 2) M(q, q) = -beta_sq / (2.0 * (2 * q - 1) * (2 * q + 3));
      if (q + 2 < N) M(q + 2, q) = beta_sq / (4.0 * (2 * q + 1) * (2 * q + 3));
      if (q - 2 >= 0) M(q - 2, q) = beta_sq / (4.0 * (2 * q - 1) * (2 * q + 1));
    }
    return M;
  }

  // Bosonic Legendre coefficients d_n = (2n+1)/beta int_0^beta D(tau) P_n(2 tau/beta - 1) dtau, by the trapezoidal rule on n_pt points
  inline nda::vector<double> fit_legendre_coeffs(int n_pt, double beta, std::function<double(double)> D0_eval, int N) {
    nda::vector<double> d_n = nda::zeros<double>(N);
    double dtau             = beta / (n_pt - 1.0);

    for (int i = 0; i < n_pt; ++i) {
      double tau    = i * dtau;
      double D      = D0_eval(tau);
      double weight = (i == 0 || i == n_pt - 1) ? 0.5 : 1.0;
      triqs::utility::legendre_generator gen;
      gen.reset(2.0 * tau / beta - 1.0);
      for (int n = 0; n < N; ++n) d_n(n) += weight * D * gen.next() * dtau;
    }
    for (int n = 0; n < N; ++n) d_n(n) *= (2.0 * n + 1.0) / beta;
    return d_n;
  }

} // namespace triqs_cthyb
