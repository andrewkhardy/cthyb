#pragma once
#include <nda/nda.hpp>
#include <triqs/utility/exceptions.hpp>
#include <algorithm>
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

  // Bosonic Legendre coefficients d_n = (2n+1)/beta int_0^beta D(tau) P_n(2 tau/beta - 1) dtau, n < N, of the
  // piecewise-linear interpolant of D through n_pt equidistant samples (end points included), computed exactly.
  //
  // In x = 2 tau/beta - 1, integrate by parts twice against the second antiderivative of P_n,
  //   Pt_n(x) = [(P_{n+2} - P_n) / (2n+3) - (P_n - P_{n-2}) / (2n-1)] / (2n+1)    (n >= 2),
  // which vanishes with its derivative at x = +-1, leaving the jumps of the interpolant's slope s at the interior nodes:
  //   int_{-1}^{1} D P_n dx = sum_i Pt_n(x_i) (s_i - s_{i-1}).
  // n = 0 is the trapezoidal sum; n = 1 has Pt_1 = (x^3/3 - x)/2 - 1/3 and the boundary term 2 s_last / 3.
  // The only error is that of the interpolant, (w dtau)^2 / 12 relative in K for a boson of frequency w, whatever n.
  inline nda::vector<double> fit_legendre_coeffs(int n_pt, double beta, std::function<double(double)> D0_eval, int N) {
    nda::vector<double> d_n = nda::zeros<double>(N);
    if (N <= 0) return d_n;
    if (n_pt < 2) TRIQS_RUNTIME_ERROR << "fit_legendre_coeffs: need at least 2 tau points, got " << n_pt;

    double const h = 2.0 / (n_pt - 1); // grid step in x
    std::vector<double> f(n_pt);
    for (int i = 0; i < n_pt; ++i) f[i] = D0_eval(i * beta / (n_pt - 1.0));

    int const m = n_pt - 2; // interior nodes x_1 .. x_{n_pt-2}, stored at k = i - 1
    std::vector<double> x(m), jump(m);
    for (int k = 0; k < m; ++k) {
      x[k]    = -1.0 + (k + 1) * h;
      jump[k] = ((f[k + 2] - f[k + 1]) - (f[k + 1] - f[k])) / h;
    }
    double const s_last = (f[n_pt - 1] - f[n_pt - 2]) / h;

    double sum = 0.5 * (f.front() + f.back());
    for (int i = 1; i < n_pt - 1; ++i) sum += f[i];
    d_n(0) = sum * h;

    if (N > 1) {
      double I1 = 2.0 / 3.0 * s_last;
      for (int k = 0; k < m; ++k) I1 += ((x[k] * x[k] * x[k] / 3.0 - x[k]) / 2.0 - 1.0 / 3.0) * jump[k];
      d_n(1) = I1;
    }

    if (N > 2) {
      // P[j] holds P_{n-2+j} at the interior nodes, j = 0..4; start at n = 2 with P_0 .. P_4
      std::vector<std::vector<double>> P(5, std::vector<double>(m));
      for (int k = 0; k < m; ++k) {
        P[0][k] = 1.0;
        P[1][k] = x[k];
        for (int l = 1; l < 4; ++l) P[l + 1][k] = ((2 * l + 1) * x[k] * P[l][k] - l * P[l - 1][k]) / (l + 1);
      }
      for (int n = 2; n < N; ++n) {
        double const a = 1.0 / (2 * n + 3), b = 1.0 / (2 * n - 1);
        double In      = 0.0;
        for (int k = 0; k < m; ++k) In += ((P[4][k] - P[2][k]) * a - (P[2][k] - P[0][k]) * b) * jump[k];
        d_n(n) = In / (2 * n + 1);
        // Shift to n + 1: drop P_{n-2}, add P_{n+3} by the Bonnet recurrence from l = n + 2
        std::rotate(P.begin(), P.begin() + 1, P.end());
        int const l = n + 2;
        for (int k = 0; k < m; ++k) P[4][k] = ((2 * l + 1) * x[k] * P[3][k] - l * P[2][k]) / (l + 1);
      }
    }

    // d_n = (2n+1)/beta * (beta/2) int_{-1}^{1} D P_n dx
    for (int n = 0; n < N; ++n) d_n(n) *= (2.0 * n + 1.0) / 2.0;
    return d_n;
  }

  // out[n] += sum_p w[p] P_n(x[p]) for n < n_l and x[p] in [-1, 1], the Legendre moments of weighted points. The Bonnet
  // recursions of all the points run side by side, one order at a time, with precomputed coefficients, so the loop over
  // the points vectorizes.
  class legendre_sums {
    public:
    explicit legendre_sums(int n_l) : a(n_l), b(n_l) {
      for (int n = 0; n < n_l; ++n) {
        a[n] = (2.0 * n + 1.0) / (n + 1.0);
        b[n] = n / (n + 1.0);
      }
    }

    template <typename T> void add(std::vector<double> const &x, std::vector<T> const &w, T *out) {
      int const n_l = a.size();
      if (x.empty() || n_l == 0) return;
      P_prev.assign(x.size(), 1.0);
      P.assign(x.begin(), x.end());
      out[0] += sum(w, P_prev);
      if (n_l > 1) out[1] += sum(w, P);
      for (int n = 1; n + 1 < n_l; ++n) {
        for (size_t p = 0; p < x.size(); ++p) {
          double const P_next = a[n] * x[p] * P[p] - b[n] * P_prev[p];
          P_prev[p]           = P[p];
          P[p]                = P_next;
        }
        out[n + 1] += sum(w, P);
      }
    }

    private:
    // sum_p w[p] v[p] in four partial sums, so that the additions do not wait on each other
    template <typename T> static T sum(std::vector<T> const &w, std::vector<double> const &v) {
      T s[4] = {};
      size_t p = 0;
      for (; p + 4 <= v.size(); p += 4)
        for (int l = 0; l < 4; ++l) s[l] += w[p + l] * v[p + l];
      for (; p < v.size(); ++p) s[0] += w[p] * v[p];
      return (s[0] + s[1]) + (s[2] + s[3]);
    }

    std::vector<double> a, b, P, P_prev;
  };

} // namespace triqs_cthyb
