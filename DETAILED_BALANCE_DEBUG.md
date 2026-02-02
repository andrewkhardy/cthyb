# Detailed Balance Issues - Debugging Guide

## Identified Issues

### 1. **Proposal Probability Asymmetry**

**Insert move (line 61):**
```cpp
mc_weight_t t_ratio = config.beta() * config.beta() / double(config.dyn_oplist.size() + 1);
```

**Remove move (line 62):**
```cpp
mc_weight_t t_ratio = double(config.dyn_oplist.size()) / (config.beta() * config.beta());
```

**Problem:** These are not exact inverses!
- Insert: `beta^2 / (n+1)` 
- Remove: `n / beta^2`

For detailed balance, we need: `T_insert * P_insert = T_remove * P_remove`

The proposal ratio should be:
- Insert: Choose 2 random times (tau1, tau2) from beta × beta, choose operator pair from M types
  - Proposal probability: `1/(beta^2 * M)`
- Remove: Choose one pair from n existing pairs
  - Proposal probability: `1/n`

**Expected ratio:** `T_remove/T_insert = n/(beta^2 * M)`

But the code has: `n/beta^2` vs `beta^2/(n+1)`, missing the factor of M (number of operator types).

### 2. **Time Ordering Issue**

**Insert move (line 34-35):**
```cpp
tau1 = data.tau_seg.get_random_pt(rng);
tau2 = data.tau_seg.get_random_pt(rng);
if (tau1 < tau2) std::swap(tau1, tau2);
```

This ensures `tau1 >= tau2`, which restricts the sampling to half of the phase space! 

**Problem:** You're only sampling ordered pairs, which means:
- You sample `beta^2/2` phase space, not `beta^2`
- The dynamic interaction term `V(tau1-tau2)` should account for this

### 3. **Dynamic Interaction Evaluation**

**Insert move (line 58):**
```cpp
double dyn_term_ratio = data.dyn_interactions[dyn_pair.f_index](double(tau1 - tau2));
```

**Remove move (line 57):**
```cpp
double dyn_term_ratio = 1.0 / data.dyn_interactions[dyn_pair.f_index](double(tau1 - tau2));
```

**Potential Issue:** For bosonic Green's functions:
- `V(tau)` for tau > 0
- `V(beta + tau) = V(-tau)` for tau < 0 (bosonic periodicity)

Since `tau1 >= tau2` is enforced, `tau1 - tau2 >= 0`, but check if the evaluation correctly handles the periodicity.

## Debugging Steps

### Step 1: Add Detailed Logging

Add a debug flag to print detailed balance checks:

```cpp
// In insert_dyn.cpp, add after line 61:
#ifdef DEBUG_DETAILED_BALANCE
std::cout << "INSERT: n=" << config.dyn_oplist.size() 
          << " tau1=" << tau1 << " tau2=" << tau2
          << " dtau=" << (tau1-tau2)
          << " V(dtau)=" << data.dyn_interactions[dyn_pair.f_index](double(tau1 - tau2))
          << " t_ratio=" << t_ratio
          << " M=" << data.dyn_op_list.size() << std::endl;
#endif
```

### Step 2: Verify Proposal Probabilities

The correct detailed balance condition:
```
P_accept(insert) / P_accept(remove) = T_propose(remove) / T_propose(insert)
```

Where:
- `T_propose(insert) = 1/(beta^2 * M)` (pick 2 times and 1 operator type)
- `T_propose(remove) = 1/n` (pick 1 pair from n)

So: `t_ratio_insert = beta^2 * M / (n+1)` ← **Missing M factor!**

### Step 3: Fix Time Ordering

**Option A:** Sample full space and use symmetry
```cpp
// Don't swap - sample full beta^2
tau1 = data.tau_seg.get_random_pt(rng);
tau2 = data.tau_seg.get_random_pt(rng);
double dtau = double(tau1 - tau2);
// Handle negative dtau with bosonic periodicity
if (dtau < 0) dtau += config.beta();
double V = data.dyn_interactions[dyn_pair.f_index](dtau);
```

**Option B:** Keep ordering but fix prefactor
```cpp
// If restricting to tau1 >= tau2, proposal is 1/(beta^2/2 * M)
mc_weight_t t_ratio = config.beta() * config.beta() / (2.0 * data.dyn_op_list.size() * double(config.dyn_oplist.size() + 1));
```

### Step 4: Check Particle-Hole Symmetry

For particle-hole symmetry at half-filling:
```cpp
// Check that <n_up> = <n_down> = 0.5
// Check that G_up(tau) = G_down(tau)
// Add measurements:
measure_density_by_flavor();
```

### Step 5: Test with Known Case

Test with `J=0` (no dynamic interaction):
- Should reduce to standard CTHYB
- Should preserve particle-hole symmetry
- Dynamic moves should have zero acceptance (since V=0)

## Recommended Fixes

1. **Fix the proposal ratio to include M:**
```cpp
// insert_dyn.cpp line 61:
mc_weight_t t_ratio = config.beta() * config.beta() * data.dyn_op_list.size() / double(config.dyn_oplist.size() + 1);
```

2. **Handle time differences consistently:**
```cpp
// Use modulo to handle tau difference properly
double dtau = double(tau1 - tau2);
// For bosonic GF: V(tau) = V(-tau) = V(beta - tau)
// Wrap to [0, beta)
while (dtau < 0) dtau += config.beta();
while (dtau >= config.beta()) dtau -= config.beta();
```

3. **Add assertions:**
```cpp
// Check detailed balance numerically in debug mode
#ifdef DEBUG_DETAILED_BALANCE
// After every insert/remove pair, verify ratios
#endif
```

## Mathematical Check

For detailed balance:
```
W(C -> C') / W(C' -> C) = exp(-beta * (E(C') - E(C)))
```

Where W includes both proposal and acceptance:
```
W = T_propose * min(1, acceptance_ratio)
```

The acceptance ratio should be:
```
acceptance = (Z'/Z) * (T_remove/T_insert)
            = (atomic_weight' / atomic_weight) * V(tau1-tau2) * (n/(beta^2 * M)) / (1/(n+1))
            = (atomic_weight' / atomic_weight) * V(tau1-tau2) * n * (n+1) / (beta^2 * M)
```

Current code has this without the M factor, which will break detailed balance!
