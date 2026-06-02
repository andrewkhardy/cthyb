import re

with open("c++/triqs_cthyb/measures/D0_corr.cpp", "r") as f:
    content = f.read()

# Fix 1: Restore the signs
old_self = """    // Self-pairs: alpha=beta, dt=0 => folded dt=0 => x=-1.  S_alpha^2=1.
    leg.reset(-1.0);
    std::vector<double> Pn_self(n_leg);
    for (int n = 0; n < n_leg; ++n) Pn_self[n] = leg.next();

    for (size_t i = 0; i < ops.size(); ++i) {
      auto const &[t1, op1] = ops[i];
      int const a            = data.linindex.at({op1.block_index, op1.inner_index});
      for (int n = 0; n < n_leg; ++n) alpha_n(a, a, n) += s * Pn_self[n];
    }"""

new_self = """    // Self-pairs: alpha=beta, dt=0 => folded dt=0 => x=-1.  S_alpha^2=1.
    leg.reset(-1.0);
    std::vector<double> Pn_self(n_leg);
    for (int n = 0; n < n_leg; ++n) Pn_self[n] = leg.next();

    for (size_t i = 0; i < ops.size(); ++i) {
      auto const &[t1, op1] = ops[i];
      int const a            = data.linindex.at({op1.block_index, op1.inner_index});
      // S_alpha * S_alpha = 1, so no sign needed here
      for (int n = 0; n < n_leg; ++n) alpha_n(a, a, n) += s * Pn_self[n];
    }"""

old_cross = """      for (size_t j = i + 1; j < ops.size(); ++j) {
        auto const &[t2, op2] = ops[j];
        int const b            = data.linindex.at({op2.block_index, op2.inner_index});

        // Fold dt into [0, beta/2]; both (tau_i - tau_j) and (tau_j - tau_i)
        // give the same |dt| so this correctly counts both orderings.
        double dt = std::abs(double(t1) - double(t2));
        if (dt > beta / 2.0) dt = beta - dt;
        double x = 2.0 * dt / beta - 1.0; // x in [-1, 0]
        leg.reset(x);

        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t val = s * leg.next(); // S_alpha = S_beta = +1
          alpha_n(a, b, n) += val;
          alpha_n(b, a, n) += val; // == alpha_n(a,a,n) += 2*val when a==b
        }
      }"""

new_cross = """      double s1 = op1.dagger ? 1.0 : -1.0;
      for (size_t j = i + 1; j < ops.size(); ++j) {
        auto const &[t2, op2] = ops[j];
        int const b            = data.linindex.at({op2.block_index, op2.inner_index});
        double s2 = op2.dagger ? 1.0 : -1.0;

        // Fold dt into [0, beta/2]; both (tau_i - tau_j) and (tau_j - tau_i)
        // give the same |dt| so this correctly counts both orderings.
        double dt = std::abs(double(t1) - double(t2));
        if (dt > beta / 2.0) dt = beta - dt;
        double x = 2.0 * dt / beta - 1.0; // x in [-1, 0]
        leg.reset(x);

        for (int n = 0; n < n_leg; ++n) {
          mc_weight_t val = s * s1 * s2 * leg.next(); // Restore operator signs!
          alpha_n(a, b, n) += val;
          alpha_n(b, a, n) += val; // == alpha_n(a,a,n) += 2*val when a==b
        }
      }"""

# Fix 2: Normalization
old_norm = """          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          q_n(a, b, n) = sum / (beta * (2.0 * n + 1.0));"""

new_norm = """          for (int p = 0; p < n_leg; ++p) sum += M(p, n) * (alpha_n(a, b, p) / norm);
          q_n(a, b, n) = sum * (2.0 * n + 1.0) / (beta * beta);"""


content = content.replace(old_self, new_self)
content = content.replace(old_cross, new_cross)
content = content.replace(old_norm, new_norm)

with open("c++/triqs_cthyb/measures/D0_corr.cpp", "w") as f:
    f.write(content)

print("Patch applied.")
