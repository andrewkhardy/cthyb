# Multiorbital dynamical-interaction extension for CTHYB — implementation notes

Branch: `multiorbital-v2` (based on `unstable`, with `multiorbital`'s work merged in).
Base commit for everything below: `48cb130` ("starting updates for multiorbital"),
already pushed to `origin/multiorbital-v2`. Everything described as "uncommitted" below
is local, on top of that commit.

## Goal

Generalize CTHYB's retarded/dynamical-interaction machinery (previously: a scalar
Jperp spin-flip channel + a block-pair D0 density-density channel, each handled either
by the stochastic double expansion or, for D0, by the analytic Lang-Firsov
resummation) to:

1. Support genuine 4-index dynamical vertices (arbitrary fermion bilinear × fermion
   bilinear), not just the hardcoded Jperp/D0 shapes — user-specified via
   `solver.add_dyn_vertex(op1, op2, coupling)`.
2. Automatically determine, per vertex, whether it can be handled by the analytic
   Lang-Firsov resummation or needs the stochastic double expansion — by symbolic
   operator algebra against `h_loc`, not by guessing from input shape.
3. Handle a general dynamical Hubbard-Kanamori Hamiltonian correctly: density-density
   terms go analytic when possible, spin-flip/pair-hopping-shaped terms go stochastic,
   and (new) even when individual orbital densities don't commute with `h_loc` (e.g.
   under real spin-flip/pair-hopping), the *total* density often still does, and can be
   handled analytically too.
4. Eventually validate all of this against a rotated-basis (bonding/antibonding)
   two-site dimer DCA test. **Not started yet.**

## What's done

### 1. Core vertex data model (`c++/triqs_cthyb/configuration.hpp`)

`dyn_vertex_t { many_body_op_t op1, op2; gf<imtime, scalar_valued> coupling; }` — a
user-facing dynamical vertex `D(tau) * op1(tau) * op2(0)`, where op1/op2 are ordinary
`many_body_operator` expressions (e.g. `c_dag('up',0)*c('down',0)`), exactly like
`h_int`. This is the common representation that D0_tau, Jperp_tau, and explicit
`add_dyn_vertex` calls all get converted into before anything else happens.

Distinct from the older, lower-level `op_desc_pair_t`/`bosonic_op_pair_t`
(block/inner-index/dagger based) which is what the stochastic double expansion
(`moves/insert_dyn.cpp`, `moves/remove_dyn.cpp`) actually consumes — `dyn_vertex_t`
gets converted down to that via `extract_bilinear` when needed.

### 2. Solver-facing API (`c++/triqs_cthyb/solver_core.hpp`)

- `inputs.dyn_vertices: std::vector<dyn_vertex_t>` alongside the existing `D0t`/`Jperpt`.
- `solver.add_dyn_vertex(op1, op2, coupling)`: register one explicit vertex. Can be
  called any number of times; combines with (doesn't replace) D0_tau/Jperp_tau.
- `Jperp_tau()` is back to a **scalar** `gf<imtime>` (single global up/down coupling,
  `{1,1}` shape) — matching ctseg's interface exactly and pre-multiorbital cthyb.
  `multiorbital`'s original merge had promoted this to `block2_gf`; that was reverted
  (see "Design decisions" below) since it's redundant with `add_dyn_vertex` for the
  general case and the user wants `Jperp_tau`/`D0_tau` to stay simple convenience
  sugar, matching ctseg 1:1 for the simple cases.
- H5 read/write for `dyn_vertices` added.

### 3. The unified pipeline (`c++/triqs_cthyb/dynamical_interactions.hpp/.cpp`, new files)

This is where essentially all the new logic lives. Pipeline, in order (see
`solver_core.cpp`'s `solve()` for the actual call sites):

1. **`collect_dyn_vertices`**: merge `inputs.dyn_vertices` (explicit) with
   `expand_D0_into_vertices`/`expand_Jperp_into_vertices` (D0_tau/Jperp_tau expanded
   into the same `dyn_vertex_t` representation) into one list.
   - `expand_D0_into_vertices`: every nonzero `(bl1,i1,bl2,i2)` entry of D0_tau becomes
     its own vertex; no orbital/spin convention needed for density-density.
   - `expand_Jperp_into_vertices`: Jperp_tau is scalar, so this only supports exactly 2
     blocks, each 1 orbital — raises a clear error otherwise, directing to
     `add_dyn_vertex` for anything more general. No more guessing a "two-block vs
     interleaved" layout (that was the old `multiorbital` design, removed).

2. **`find_conserved_density_combinations` + `recover_conserved_density_groups`** (see
   "Conserved-density-combination recovery" below): finds every linear combination of
   orbital densities that commutes with `h_loc` (not just `N_total` — e.g. total
   spin-up density and total spin-down density separately, for a standard Kanamori
   `h_loc`), then rescues any *completely* user-specified group of vertices (diagonal
   self-terms included) that exactly reconstructs a coupling to one of them.

3. **`classify_dyn_vertices`**: splits whatever's left into `.lang_firsov` and
   `.stochastic`. A vertex is Lang-Firsov-eligible only if both op1 and op2 are number
   operators (`is_density_bilinear`) *and* both individually commute with `h_loc`,
   checked via plain symbolic operator algebra: `(op*h_loc - h_loc*op).is_almost_zero()`.
   Deliberately **not** implemented via `atom_diag::quantum_number_eigenvalues[_checked]`
   — that requires diagonalizing `h_loc` first, gives false negatives on unresolved
   degenerate eigenspaces, and can segfault TRIQS if fed a non-commuting operator as a
   `qn_vector` hint (real upstream bug, reproducer at
   `/tmp/.../minimal_repro_atom_diag_segfault.py` if it still exists — not filed
   upstream yet).
   - When `params.lang_firsov == false`, eligibility is never evaluated — everything
     goes stochastic. This is still the master on/off switch, used for debugging /
     forcing a stochastic-only comparison run.

4. **`apply_lang_firsov_shift`**: subtract the K'(0) static part of each analytic
   vertex's coupling from `h_loc` (including any vertices recovered by
   `recover_conserved_density_groups`, moved in before this runs — see
   "Conserved-density-combination recovery" below), before `h_diag` is built. At
   `verbosity>=2`, also prints the aggregate before/after density-density interaction
   matrix (as in CTSEG) — this was explicitly requested to be kept.

5. **`build_K_n`**: build the `K_n[a][b][:]` Legendre-coefficient kernel consumed by
   `qmc_data::compute_lang_firsov_ratio`, from the same merged vertex list.

6. **`fold_into_stochastic_catalog`**: convert whatever's left in `.stochastic` into
   `bosonic_op_pair_t`/`dyn_op_list`/`dyn_interactions`, consumed by
   `moves/insert_dyn.cpp`/`remove_dyn.cpp`, exactly as before.

`has_dyn_interactions` (gates whether the stochastic moves/measure get registered at
all) is now simply `!dyn_op_list.empty()`.

### 4. Conserved-density-combination recovery (rewritten 2026-08-14)

The original "total-density decomposition" design was mathematically unsound; see the
2026-08-14 dated entry below for the full correction and why.

**Motivation**: the per-vertex check in `classify_dyn_vertices` requires each
individual `n_a` to commute with `h_loc`. That's correct but conservative: a real
Hubbard-Kanamori `h_loc` with spin-flip and/or pair-hopping does *not* conserve
individual orbital densities. What it *does* conserve is certain **linear combinations**
of them — total charge `N_total = sum_a n_a` always, and for a standard Kanamori
`h_loc` (2-orbital case checked directly), total spin-up density `N_up = sum_a n_{a,up}`
and total spin-down density `N_down = sum_a n_{a,down}` *individually* (equivalently,
`N_total` and total `S_z` — the spin-flip and pair-hopping terms, correctly read as
4-operator processes, always move a particle *within* a fixed spin channel or move an
up+down pair together, never converting spin, so each spin channel's total count is
separately conserved). A dynamical vertex set that, together, amounts to a coupling to
one of these conserved combinations should be Lang-Firsov-eligible even though every
individual vertex fails the per-vertex check.

**The right general criterion, and why "does a summed operator commute" alone is not
enough**: `K_n[a][b]` is indexed per *individual* orbital, and `compute_lang_firsov_ratio`
dresses every pair of fermion-operator insertions (at orbitals `a` and `b`) with a
phase from `K_n[a][b]` — there is no assumption that `K_n` is diagonal, but each
individual density `n_a` used this way needs to be a piecewise-constant, deterministic
function of insertion history between hybridization events for the phase formula to be
exact. That's exactly what "`n_a` commutes with `h_loc`" means operationally.
Substituting a *different*, weaker check — does the naive sum of the vertices' own
orbitals commute — was the original design's mistake (see the dated entry): it doesn't
verify that the specific combination the K_n matrix would end up representing is
actually one that's conserved, only that some unrelated aggregate is. The fix is to
find the *actual* conserved subspace directly and require an *exact* match against it.

**`find_conserved_density_combinations`** (`dynamical_interactions.cpp`): computes the
space `{ c in R^M : [sum_a c_a n_a, h_loc] = 0 }` (`M` = total orbital count) as a
genuine linear-algebra problem, not a guess. The commutator is linear in `c`, so this
is exactly the nullspace of the linear map `c -> sum_a c_a [n_a, h_loc]`. Computed by:
expanding each `[n_a, h_loc]` via ordinary symbolic operator algebra (same primitive
`classify_dyn_vertices` already uses), collecting every distinct monomial appearing
across all `M` commutators as rows of a real coefficient matrix (real and imaginary
parts of each coefficient as separate rows, so this is correct whether `h_scalar_t` is
real or complex), and taking that matrix's nullspace via SVD (`nda::linalg::svd`).
`n_a` itself must be built from `fundamental_operator_set`'s own `indices_t` for that
linear position (`fundamental_operator_set::data_t(fops)[a]` + `many_body_op_t::make_canonical`)
— **not** from `linindex`'s `(block_index, inner_index)` key directly, which uses
`gf_struct`'s block *position*, not its name; building `c_dag<h_scalar_t>(block_index,
inner_index)` from those raw ints silently constructs an operator on a fundamental mode
unrelated to `h_loc`'s actual algebra, making every commutator trivially (and wrongly)
zero — a real bug hit and fixed while implementing this.

**`recover_conserved_density_groups`** (`dynamical_interactions.cpp`): runs *after*
`classify_dyn_vertices`, only on what it rejected (`classified.stochastic`) — this
matters, see below. Groups density-bilinear rejected vertices by shared coupling curve
(`gf_close`), then for each group:
1. Requires **every** ordered pair `(a,b)` among the touched orbitals — **including the
   diagonal `a==b` self-terms** — to already exist as its own vertex sharing that
   coupling. Nothing is inferred or filled in: a missing diagonal self-term is never
   silently added, even when the off-diagonal part alone would numerically suggest one.
   This is the point the original design got wrong (see dated entry) — the reason it
   matters isn't just bookkeeping: for a purely off-diagonal specification (no diagonal
   given at all), the interaction is genuinely `sum_{a!=b} D(tau) n_a(tau) n_b(0)`, a
   *different* physical object than `D(tau) N_total(tau) N_total(0)` (which also
   includes the `n_a(tau)n_a(0)` self-correlator terms) — silently adding the missing
   diagonal changes the physics being asked for, however numerically tempting.
2. Builds the group's target coupling matrix (all touched-pair entries, `1` in units of
   the shared coupling curve) and fits it as `sum_ij Gamma_ij * O_i(a) * O_j(b)` over
   the conserved combinations `O_i` found above (`Gamma` symmetric, so both "coupled to
   a single combination" and "coupled to a linear combination of several" are covered;
   cross terms are valid too, not just each `O_i` alone — any two conserved density
   combinations automatically commute with each other, since number operators for
   different modes always do, regardless of `h_loc`). SVD-based least-squares
   (pseudo-inverse), robust to a rank-deficient design.
3. Only if the fit is *exact* (residual ~0, not merely small) does the group move from
   `classified.stochastic` to `classified.lang_firsov` — as-is, no reconstruction
   needed, since every vertex was already fully user-specified. The ordinary,
   unmodified `apply_lang_firsov_shift`/`build_K_n` handle the rest, exactly as if
   `classify_dyn_vertices` had accepted it directly.

Running recovery only on `classify_dyn_vertices`'s leftovers (not on the full vertex
list, as the original design did) also fixes a second, independent problem the
diagonal-overwrite bug exposed: a vertex set that `classify_dyn_vertices` *already*
handles correctly on its own (e.g. `spin_spin.cpp`'s D0 terms — `h_loc` there is pure
density-density, so every individual `n_a` already commutes) never reaches the recovery
step at all now, so it can't be second-guessed or reinterpreted by it.

**Validated**: `test/python/kanamori_dyn.py` — a full static Kanamori `h_loc`
(spin-flip *and* pair-hopping) with a phonon coupled to total density, specified
*completely* (every off-diagonal pair via `kanamori_dynamical_vertices`, every diagonal
self-term explicitly via `add_dyn_vertex`, all sharing one coupling curve) — is
correctly recovered: `find_conserved_density_combinations` finds 2 combinations
(`N_up`, `N_down`), the fit against the 16-entry (4 orbitals x 4 orbitals) target
matrix succeeds with `gamma=[2,2,2]` and residual `~1e-16`, and all 16 vertices move to
the analytic path. Dropping the diagonal terms (the original, incomplete
specification) correctly falls back to fully stochastic instead — verified both
directions. `test/python/kanamori_dyn_selfconsistency.py` cross-checks the analytic
result against forced-`lang_firsov=False` (fully stochastic) on the identical,
completely-specified setup: `G_l` agrees within tolerance (`~0.02` at 200k/300k
cycles), average sign stays near 1.0 in both. See the dated entry for the second real
bug found while getting this working (`nda::matrix` scalar assignment).

**Current scope / explicit limitation**: still requires an *exact* fit — no partial
decomposition when a group's couplings only share a *common part* (e.g. Kanamori's `U`
and `U'` being different but comparable, or a diagonal specified with a *different*
coupling than the off-diagonal). This is the confirmed necessary next generalization,
not just a nice-to-have: the eventual target is a full dynamical interaction built from
ab-initio GW data, where the total-density channel is expected to *dominate* the
retarded coupling, with non-uniform corrections handled stochastically on top — this
session's validated example is the *fully representable* limit of that picture; the
partial/residual generalization is what's needed for the realistic (non-uniform)
case. See "What's left" item 3.

### 5. Python convenience layer

- `python/triqs_cthyb/dynamical_interactions.py` (new):
  `kanamori_dynamical_vertices(solver, spin_names, orb_names, U=None, Uprime=None,
  J_hund=None, spin_flip=True)` — builds the dynamical generalization of
  `triqs.operators.util.hamiltonians.h_int_kanamori`'s density-density and spin-flip
  terms (same conventions/signs, term-for-term), registering each via
  `solver.add_dyn_vertex(...)`. Accepts `U`/`Uprime`/`J_hund` as either a single `Gf`
  (broadcast to every orbital pair) or a `dict[(a1,a2)] -> Gf` for per-pair couplings.
  Auto-converts `(1,1)` matrix_valued Gfs to the `scalar_valued` shape
  `add_dyn_vertex` actually needs (`_as_scalar_gf`).
  - **Does not support pair-hopping** — `add_dyn_vertex` requires each side to reduce
    to a single fermion bilinear (one creation, one annihilation); pair-hopping's
    `c^dag_a c^dag_b` is a pair-creation operator, not a bilinear. Representing it
    would need a genuinely different vertex type (`op_desc_pair_t`,
    `moves/insert_dyn.cpp`, `moves/remove_dyn.cpp` all assume one creation + one
    annihilation per side) — not attempted, and deliberately no `pair_hopping`
    argument exists so this isn't silently missing.
  - Registered in `python/triqs_cthyb/__init__.py`'s `__all__`.

### 6. Default behavior change

`params.lang_firsov` default flipped from `false` to `true`
(`c++/triqs_cthyb/parameters.hpp`). Justified because the new per-vertex auto-routing
makes `true` **unconditionally safe**: anything ineligible automatically falls back to
stochastic (no more hard error on any ineligible channel, which is what the old
`true` used to do). So "use Lang-Firsov wherever possible" is now the default, and
`false` remains available purely to force pure-stochastic (e.g. for debugging, or
comparing the two methods against each other, exactly as done in the validation above).

### 7. Sign-convention bug found and fixed (real, pre-existing, not introduced by this work)

`benchmark/dynamic_int/ctseg_spin_spin.py` had an **unresolved merge conflict**
between HEAD (unstable) and `multiorbital` that was a genuine physics disagreement:
opposite signs for the D0 Sz*Sz decomposition (same-spin +0.25 vs -0.25). Verified via
physical derivation (H_int = g(n_up-n_down)(b+b^dagger) -> same-spin=+, opposite=-)
and cross-checked against `test/c++/spin_spin.cpp` and `test/python/spin_spin.py`
(both sides of *that* file's own conflict agree with the same convention) that HEAD's
convention is correct; resolved `ctseg_spin_spin.py` and `benchmark/dynamic_int/spin_spin.py`
using it. `benchmark/dynamic_int/multiorb_spin_spin.py` (a benchmark script, not
resolved from a conflict, apparently written earlier this session) had the *wrong*
(flipped) convention and its own inline comment contradicted its own code — **fixed
this session** (see the dated entry below).

### 8. Validated end-to-end against ctseg (real solver cross-check)

With `lang_firsov=True`, single-orbital spin-spin model, same physical inputs
(`ctint.ref.h5` bath) fed to both `triqs_ctseg` and this cthyb: both give average sign
**exactly 1.0**; `max|G_ctseg - G_cthyb|` shrinks from 3.5 (2k cycles) to 0.17 (100k
cycles) — same convergence signature as the total-density validation above. This
confirms the D0 sign convention, the Lang-Firsov pipeline, and the scalar Jperp_tau
reversion are all correct for the case ctseg can actually check (single orbital only —
ctseg cannot do multi-orbital, hence the total-density decomposition above couldn't be
cross-checked the same way). Being single-orbital, this also never exercises
`recover_conserved_density_groups` at all (there is no off-diagonal pair to form a
group from) — see the `kanamori_dyn`/`kanamori_dyn_selfconsistency` tests (see
"Conserved-density-combination recovery" above) for the first real exercise of the
multi-orbital conserved-combination path.

## Session update (2026-08-13)

**Superseded 2026-08-14** — the "total-density decomposition" mechanism this entry
describes fixing was itself found to be unsound the next day and replaced entirely
(see the 2026-08-14 entry below and "Conserved-density-combination recovery" above).
Left as-is for the historical record of what the diagonal-overwrite bug actually was
and how it was first (incompletely) addressed.

- **Diagonal-overwrite bug found and fixed** in the total-density decomposition (see
  the revised "Diagonal correction" subsection above for the mechanism and why the old
  approach was wrong). `apply_total_density_shift`/`apply_total_density_kernel` deleted;
  `find_total_density_decomposition`'s absorbed group vertices now flow through the
  ordinary `apply_lang_firsov_shift`/`build_K_n` instead.
- **`test/c++/spin_spin.ref.h5` and `test/python/spin_spin.ref.h5` regenerated** against
  the fixed code. Both `spin_spin` and `Py_spin_spin` ctests pass again, deterministically
  (both already had fixed seeds — `23488` in the C++ test, `123*rank+567` in the Python
  one — no seed changes were needed).
- **`benchmark/dynamic_int/multiorb_spin_spin.py`'s D0 sign bug fixed** (see item 7
  above) — same-spin/opposite-spin signs now match the verified-correct convention.
- **New Hubbard-Kanamori dynamical-interaction example, landed as two real tests**
  (see "What's left" item 5): static `h_int` carries the full Kanamori structure (`U`,
  `U'=U-2J`, `J_hund`, spin-flip *and* pair-hopping); the dynamical part is a single
  boson coupled uniformly to total density across every spin-orbital, via
  `kanamori_dynamical_vertices(solver, spin_names, orb_names, U=Q_tau, Uprime=Q_tau,
  spin_flip=False)` (passing the same `Q_tau` for both slots is what makes
  `find_total_density_decomposition` accept the group). This is the *first* real
  exercise of `kanamori_dynamical_vertices` anywhere in the repo, and the first case
  where individual orbital densities are confirmed *not* to commute with `h_loc` (real
  spin-flip + pair-hopping) while `N_total` does — exactly the scenario the
  total-density decomposition exists for.
  - `test/python/kanamori_dyn.py` (`Py_kanamori_dyn`): deterministic, `n_cycles=5000`,
    checked in against `kanamori_dyn.ref.h5`. Confirmed via a `verbosity=3` diagnostic
    run: `Total number of dynamical interaction terms: 0` and 12 Lang-Firsov K'(0)
    shifts printed (all ordered pairs among the 4 spin-orbitals) — the decomposition is
    genuinely firing, nothing silently fell back to per-vertex or stochastic handling.
  - `test/python/kanamori_dyn_selfconsistency.py` (`Py_kanamori_dyn_selfconsistency`):
    `lang_firsov=True` (200k cycles) vs. forced `lang_firsov=False` (300k cycles) on the
    identical setup, compared via `G_l` (Legendre — raw `G_tau` has large single-bin
    binning noise at this vertex count/statistics that has nothing to do with physics,
    confirmed by hand: it shrinks in lockstep with the `G_l` comparison as statistics
    increase). Average sign stayed close to `1.0` in *both* runs (`0.9999`/`0.9998`) —
    the severe pure-stochastic sign problem documented in item 4 below did **not**
    reproduce for this multi-orbital setup. `max|G_l_lf - G_l_stoch|` shrank from
    ~0.1-0.2 at 20k/20k cycles to ~0.015-0.03 at 200k/300k cycles — clean convergence,
    ~2 minutes total wall-clock, well within CI-reasonable time, so this was promoted to
    a registered, CI-gating test rather than kept as a standalone script.
  - Neither test would have caught the diagonal-overwrite bug above: the coupling here
    is uniform with no independently-specified diagonal vertex to conflict with, so
    `apply_total_density_kernel`'s old overwrite would have been a no-op in this
    specific case.

## Session update (2026-08-14): the total-density decomposition was unsound, replaced

**What went wrong, precisely**: yesterday's fix (previous entry) addressed the
diagonal-overwrite symptom, but the underlying eligibility criterion was still wrong.
`find_total_density_decomposition` licensed a group of vertices for Lang-Firsov by
checking whether `N_total` (the sum of exactly the orbitals the group's own vertices
happened to touch) commutes with `h_loc` — but the vertices, once merged into the
ordinary per-vertex `lang_firsov` list, get processed *individually*, each contributing
its own `(a,b)` entry to `K_n`. That's an "orbital-resolved" treatment, and for it to
be exact, each individual `n_a`, `n_b` needs to itself be a piecewise-constant function
of insertion history between hybridization events — i.e. needs to individually commute
with `h_loc`. Checking that a summed/aggregate operator commutes instead doesn't verify
that; it was a plausible-looking but unsound substitute.

This was caught by the user, who pushed back after seeing the log line "0 analytic, 12
stochastic" turn into "12 analytic, 0 stochastic" for `kanamori_dyn.py`'s off-diagonal-
only Kanamori coupling: *"I'm confused how a spin-flip term or specific density term
commutes in Kanamori Hamiltonian? that seems wrong."* They were right — no individual
density term commutes there; the group mechanism was exploiting the aggregate `N_total`
commuting as a loophole around checking the operators it actually dresses individually,
and that loophole isn't valid. Correct references on the actual condition (Lang-Firsov
requires the *specific coupled operator* to commute with `h_loc`, not merely *some*
operator): I. G. Lang & Yu. A. Firsov, *Zh. Eksp. Teor. Fiz.* **43**, 1843 (1962); P.
Werner & A. J. Millis, *PRL* **99**, 146404 (2007) and *PRL* **104**, 146401 (2010); E.
Gull et al., *Rev. Mod. Phys.* **83**, 349 (2011); Y. Nomura, S. Sakai, M. Capone, R.
Arita, *Sci. Adv.* **1**, e1500568 (2015), Supp. §D (non-density-type couplings fall
outside Lang-Firsov); K. Steiner, Y. Nomura, P. Werner, *PRB* **92**, 115123 (2015)
(the hybridization + weak-coupling-in-J workaround this limitation motivated).

**The corrected mechanism**: see "Conserved-density-combination recovery" above (full
rewrite of section 4) — `find_conserved_density_combinations` finds the *actual*
conserved subspace of density combinations via linear algebra (nullspace of
`[sum_a c_a n_a, h_loc]`, not a guessed `N_total`), and `recover_conserved_density_groups`
only promotes a group when it is *exactly* representable in that subspace *and*
completely, explicitly specified by the user (diagonal self-terms included — no
silent inference of a missing Holstein term). This is a strictly stronger, sound
generalization: it also naturally finds combinations beyond `N_total` (e.g. `N_up`/
`N_down` separately, for a standard 2-orbital Kanamori `h_loc` — equivalent to `N_total`
and total `S_z`), which the old design would never have looked for.

**Two real implementation bugs found and fixed while getting the new mechanism working**
(both via direct debugging against `kanamori_dyn.py`'s setup, not just derivation):
1. `find_conserved_density_combinations` initially built `n_a` from `linindex`'s
   `(block_index, inner_index)` key directly (`c_dag<h_scalar_t>(block_index,
   inner_index)`) — but `block_index` there is `gf_struct`'s block *position* (an int),
   not its name, so this silently constructed an operator on a fundamental mode
   unrelated to `h_loc`'s actual algebra. Every commutator came out trivially zero (0
   conserved combinations found for a Kanamori `h_loc` that obviously has some).
   Fixed: build `n_a` from `fundamental_operator_set`'s own `indices_t` for that linear
   position instead (`fundamental_operator_set::data_t(fops)[a]` +
   `many_body_op_t::make_canonical(dagger, indices)`).
2. The least-squares fit (`recover_conserved_density_groups`'s `fit_and_residual`)
   initialized its target matrix via `nda::matrix<double> target(n,n); target = 1.0;`,
   intending an all-ones matrix. `nda::matrix`'s scalar assignment means "scalar times
   the identity matrix", not element-wise fill (confirmed by printing the flattened
   target and seeing the identity pattern) — a real, non-obvious library-semantics trap
   distinguishing `nda::matrix` from a plain array type. Fixed with an explicit
   double loop. (Note for future code in this file: any other `nda::matrix = <nonzero
   scalar>` should be treated with suspicion; `= 0.0` is safe since both readings agree
   there, which is presumably why this didn't surface earlier in `bare_density_matrix`'s
   `U_matrix = 0.0`/`mu_vec = 0.0`.)

Both bugs were caught by hand-deriving the expected result (`gamma=[2,2,2]` for the
`kanamori_dyn.py` setup, worked out on paper from the `N_up`/`N_down` basis) and adding
temporary `std::cerr` tracing to compare against actual runtime values — the mismatch
(`gamma=[1,~0,1]`, wrong) pointed straight at the target-matrix bug once the conserved
combinations themselves were confirmed correct. Temporary tracing was removed before
landing; `solver_core.cpp` keeps one `verbosity>=2` production line reporting how many
conserved combinations were found and how many vertices were recovered by them.

**Test changes**: `test/python/kanamori_dyn.py` now explicitly registers the 4 diagonal
self-terms (`add_dyn_vertex(n(s,a), n(s,a), coupling)` for every spin-orbital) alongside
the off-diagonal `kanamori_dynamical_vertices(...)` call — required for the group to be
*completely* specified under the corrected (stricter, sound) criterion; without them it
now correctly falls back to fully stochastic (verified both ways). `kanamori_dyn.ref.h5`
regenerated accordingly (`16 analytic, 0 stochastic`, average sign `0.9996`).
`test/python/kanamori_dyn_selfconsistency.py` updated the same way; still passes
(`G_l` agreement `~0.02` at 200k/300k cycles, both average signs `~0.9998-0.9999`).

## Key files touched this session (uncommitted, on top of `48cb130`)

- `c++/triqs_cthyb/dynamical_interactions.hpp` (162 lines) / `.cpp` (442 lines) — most
  of the new logic, see above.
- `c++/triqs_cthyb/solver_core.cpp`/`.hpp` — rewired `solve()` to call the pipeline
  above instead of the old inline dual-mode Jperp/D0 construction; `Jperp_tau` reverted
  to scalar.
- `c++/triqs_cthyb/parameters.hpp` — `lang_firsov` default `false` -> `true`.
- `python/triqs_cthyb/dynamical_interactions.py` (new, 124 lines), wired into
  `__init__.py`.
- `benchmark/dynamic_int/ctseg_spin_spin.py`, `benchmark/dynamic_int/spin_spin.py` —
  merge conflicts resolved (D0 sign convention + scalar Jperp_tau).
- `test/c++/spin_spin.cpp` — `Jperp_tau()(0,0) = J0t` -> `Jperp_tau() = J0t` (scalar).

## Build / install / test commands

This repo builds against TRIQS installed at `/home/andrewhardy/Documents/CCQ/unstable/installation`,
and installs itself to its own prefix, `/home/andrewhardy/Documents/CCQ/installation`
(**different path** — always set `PYTHONPATH` explicitly when testing from Python, the
default `import triqs_cthyb` will pick up a stale/different build otherwise):

```bash
cd /home/andrewhardy/Documents/CCQ/cthyb_dyn/build
cmake --build . --target triqs_cthyb_c -j 20     # C++ lib only, fast iteration
cmake --build . --target install -j 20           # also rebuilds+installs python bindings
cmake --build . --target spin_spin -j 20 && ./test/c++/spin_spin   # the C++ regression test

# Python, after install:
PYTHONPATH=/home/andrewhardy/Documents/CCQ/installation/lib/python3.12/site-packages:$PYTHONPATH python3 your_script.py
```

Real-compiler sanity check pattern used throughout (IDE diagnostics were repeatedly
unreliable this session — always verify with the actual compiler, not the panel):
extract the exact flags for a file from `build/compile_commands.json`, swap `-c -o X`
for `-fsyntax-only`, run directly.

## What's left / next steps (in the order discussed)

### 1. Markdown doc — this file. Done.

### 2. Merge-conflict cleanup (mostly done)

**Important finding**: three files were committed into `48cb130` with **literal,
unresolved git conflict markers still in them** — i.e. they were broken (invalid
Python/shell/JSON) in the repo as pushed to `origin/multiorbital-v2`:
- `test/python/spin_spin.py` — 21 marker lines (7 conflict blocks). **Resolved.**
  Took HEAD's side throughout (params U=2.0/l=0.5, Bethe-lattice bath, scalar
  `Jperp_tau` assignment, tail-fit solve params) — verified via
  `git show unstable:test/python/spin_spin.ref.h5` vs `git show multiorbital:...`
  (multiorbital's side doesn't even have this file) that the checked-in reference
  file matches HEAD/unstable's parameters, not multiorbital's (opposite situation from
  the C++ test). Syntax-checked with `py_compile`.
- `benchmark/dynamic_int/run_benchmark.sh` — 3 marker lines (1 block). **Resolved.**
  Personal SLURM launcher, no correctness question — merged both sides' run commands
  as commented alternatives, one left active.
- `benchmark/dynamic_int/CTHYB_Benchmarking.ipynb` — 33 marker lines (11 blocks).
  **Deferred at the user's request** ("ignore this notebook for now, we can fix it").
  Still has literal conflict markers baked in, currently invalid JSON. Both sides are
  recoverable if needed: `git show unstable:benchmark/dynamic_int/CTHYB_Benchmarking.ipynb`
  (28 cells, valid JSON) vs `git show multiorbital:benchmark/dynamic_int/CTHYB_Benchmarking.ipynb`
  (44 cells, valid JSON) — per the policy below, unstable's side is very likely the
  right pick when this gets picked back up, not multiorbital's (an earlier attempt at
  this defaulted to multiorbital's side by "more complete" reasoning and was
  explicitly corrected).

**Status update (this session, 2026-08-13)**: the notebook is currently valid JSON
with no literal conflict markers present — so the state described above (broken/invalid
JSON) is no longer accurate. However, its *content* was never manually reviewed this
session (an automatic checkpoint commit bundled in a notebook change that predated this
session's own work, from an unknown prior edit), so which side (or what merge) actually
ended up in the file is unverified — needs a real content review against the policy
below, not assumed correctly resolved just because it parses.

**New conflict-resolution policy established this session** (saved to memory,
`feedback_merge_conflict_policy.md`): default to **unstable's side** for
multiorbital/unstable conflicts, unless there's a specific, verified reason to prefer
multiorbital's (e.g. reference/test data that provably only exists on multiorbital's
branch history, checked via `git log --oneline <branch> -- <file>` — this is why
`test/c++/spin_spin.cpp` correctly kept multiorbital's parameters, verified before
this policy was stated). If in doubt whether an already-resolved file should be
redone under this policy, ask rather than silently redo it.

`benchmark/dynamic_int/multiorb_spin_spin.py`'s wrong D0 sign convention (see point 7
above) was not a merge conflict (this file isn't in either branch's history, appears to
have been written fresh during an earlier part of this session), just carried the same
bug independently — **fixed this session** (see the dated entry below).

### 3. Partial/residual conserved-combination decomposition (confirmed necessary next step, not merely next-in-queue)

Generalize `recover_conserved_density_groups` beyond the exact-fit-only case: given a
group of density vertices whose coupling matrix (in the conserved-combination basis
`{O_i}` from `find_conserved_density_combinations`) does *not* fit exactly (e.g.
Kanamori's `U` and `U'` being different but comparable, so a coupling to `N_total`
alone gets close but not exact), project onto the nearest representable point in
`span{O_i(a)O_j(b)+O_j(a)O_i(b)}` anyway, apply *that* analytically, and feed only the
*residual* (`target - fit`, as an actual per-pair coupling) to the stochastic path —
instead of either the whole thing (current all-or-nothing fallback) or nothing.
Confirmed by the user as necessary, not optional: the eventual target is a full
dynamical interaction built from ab-initio GW data, where the total-density channel is
expected to dominate, with non-uniform corrections handled stochastically on top —
exactly this mechanism, generalized. The exact-fit machinery landed 2026-08-14 (see
"Conserved-density-combination recovery" above) is most of the way there already: the
projection/least-squares fit (`fit_and_residual`) already computes the residual, it's
just discarded (group rejected) instead of being fed onward when nonzero.

Open design question, not yet resolved: for the *diagonal* entries specifically, when
no self-term was given at all (the common case — see `kanamori_dyn.py`), is "residual
= 0 at that position" (leave the diagonal alone, add nothing) or "residual = fit's
implied value" (silently add a Holstein-like self-term) the right default? The
"infer less" principle argues for the former (never invent terms the user didn't
specify) — but that means the diagonal is asymmetric relative to the (fitted, then
subtracted) off-diagonal residual, which needs to be handled correctly in the
stochastic-catalog construction, not just noted. Needs a well-defined, provably-correct
treatment, not a heuristic.

**Naming reminder** (explicit user requirement): no unicode/math symbols anywhere in
code — plain descriptive English identifiers only (`total_density_coupling`,
`residual_coupling`, etc., not `D_bar`/`ΔD`/similar).

### 4. Pure-stochastic sign problem (deprioritized, "not terrible" expected)

`test/c++/spin_spin.cpp` with `lang_firsov=False` (or now, since the default flipped,
forced explicitly) has a severe sign problem (average sign ~0.009, or ~-0.0002 on the
pre-redesign code — both effectively noise) against the existing `spin_spin.ref.h5`
reference. Root cause **not yet found** — ruled out so far: not a regression from this
session's redesign (pre-redesign code fails just as badly, actually worse); not the D0
sign convention (validated correct via ctseg above); not the Jperp scalar-vs-block2_gf
question (bit-identical results either way, confirmed empirically). Likely candidates
not yet checked: `spin_spin.ref.h5` may simply be stale (generated by a materially
different codebase state, given `multiorbital` forked 237 commits back from current
`unstable`); or a genuine, independent bug in `moves/insert_dyn.cpp`/`remove_dyn.cpp`'s
weight formula. User's expectation: "the pure stochastic case will have a sign
problem and we can't magically make that go away... I don't think it should be
terrible" — so some sign problem is expected and fine, but the current magnitude
(effectively zero, noise-dominated) seems too severe to just be intrinsic difficulty.
Deprioritized relative to the Lang-Firsov-path work above, revisit later.

Cross-reference: the `kanamori_dyn_selfconsistency` test (dated entry below) exercises
the *other* direction of this same code (`insert_dyn`/`remove_dyn` forced on for a
multi-orbital dynamical interaction) and empirically did **not** reproduce a severe
sign problem there (average sign stayed close to 1.0 at 300k cycles) — worth keeping in
mind as a data point when this item is revisited, though the two setups differ enough
(single- vs. multi-orbital, different coupling shape) that it doesn't resolve the
single-orbital case's root cause.

### 5. Validation / test suite

- **Done**: `test/python/kanamori_dyn.py` (deterministic, `lang_firsov=True` only,
  `n_cycles=5000`, checked in against `kanamori_dyn.ref.h5`, registered as
  `Py_kanamori_dyn` — full Kanamori `h_loc` plus a *completely* specified coupling to
  total density, off-diagonal vertices from `kanamori_dynamical_vertices` and explicit
  diagonal self-terms via `add_dyn_vertex`) and `test/python/kanamori_dyn_selfconsistency.py`
  (physics validation — `lang_firsov=True` vs. forced `lang_firsov=False` on the
  identical setup, compared via `G_l`, registered as `Py_kanamori_dyn_selfconsistency`,
  ~2 minutes) — both are real, CI-gating tests exercising the multi-orbital
  conserved-density-combination recovery (see "Conserved-density-combination recovery"
  above for what they do and don't cover). The earlier scratchpad scripts this bullet
  used to point to (`total_density_validation.py`, `kanamori_dyn_smoke.py`,
  `kanamori_full_static_test.py`) are superseded by these.
- `spin_spin.ref.h5` (both `test/c++/` and `test/python/`) regenerated against the
  fixed dynamical-interaction code — **unrelated** to the
  pure-stochastic sign problem in item 4 above (that's about explicitly forcing
  `lang_firsov=False`, which neither `spin_spin.cpp` nor `spin_spin.py` do by default;
  this regeneration is the ordinary, default `lang_firsov=True` path). Item 4's own
  regeneration is still pending its own investigation — don't conflate the two.
- Eventually: the rotated-basis (bonding/antibonding) two-site dimer DCA test from the
  original plan (goal 4 at the top) — not started, lowest priority, exploratory.
- Eventually: compare against CTINT for the multi-orbital total-density case, once
  that's set up (explicitly deferred by the user for now — "ignore benchmarking
  against CTINT until we have this setup").

### 6. Present the full diff for review

Once the above settles, walk through the complete diff against `48cb130` (or against
`unstable` for the full picture) with the user before considering this mergeable.

## Design principles established this session (apply going forward)

- **"Infer less, make the user specify more so that it is correct."** Driving
  principle behind removing the old two-block/interleaved Jperp guessing, requiring
  exact completeness+uniformity for the total-density decomposition rather than a
  best-effort heuristic, and erroring loudly (not silently falling back) when an
  interface's assumptions aren't met (e.g. `expand_Jperp_into_vertices` on >2 blocks).
- **Always verify with the real compiler**, not IDE diagnostics — the panel gave
  spurious errors multiple times this session.
- **No unicode/math symbols in code** — plain English identifiers, however verbose.
- **Correctness over efficiency for anything silent.** The total-density decomposition
  is only ever applied when it's provably exact (uniform + complete + commutes with
  h_loc); never a heuristic that could quietly change physics.
- User approves incremental edits even after accepting an overall plan — don't batch
  large rewrites without checking in first on anything non-mechanical.
