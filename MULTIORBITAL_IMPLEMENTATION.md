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

2. **`find_total_density_decomposition`** (new, see "Total-density decomposition"
   below): looks for a "sufficiently symmetric" subgroup of density vertices — all
   sharing the exact same coupling `D(tau)`, covering every off-diagonal orbital pair
   completely — and pulls it out for special handling as a single `N_total` coupling.

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

4. **`apply_lang_firsov_shift`** + **`apply_total_density_shift`**: subtract the K'(0)
   static part of each analytic vertex's coupling from `h_loc`, before `h_diag` is
   built. At `verbosity>=2`, also prints the aggregate before/after density-density
   interaction matrix (as in CTSEG) — this was explicitly requested to be kept.

5. **`build_K_n`** + **`apply_total_density_kernel`**: build the `K_n[a][b][:]`
   Legendre-coefficient kernel consumed by `qmc_data::compute_lang_firsov_ratio`.

6. **`fold_into_stochastic_catalog`**: convert whatever's left in `.stochastic` into
   `bosonic_op_pair_t`/`dyn_op_list`/`dyn_interactions`, consumed by
   `moves/insert_dyn.cpp`/`remove_dyn.cpp`, exactly as before.

`has_dyn_interactions` (gates whether the stochastic moves/measure get registered at
all) is now simply `!dyn_op_list.empty()`.

### 4. Total-density decomposition (new, most recent work)

**Motivation**: the per-vertex check in `classify_dyn_vertices` requires each
individual `n_a` to commute with `h_loc`. That's correct but conservative: a real
Hubbard-Kanamori `h_loc` with spin-flip and/or pair-hopping does *not* conserve
individual orbital densities, even though it always conserves the *total* density
`N_total = sum_a n_a` (both spin-flip and pair-hopping just move particles between
orbitals). So a dynamical density-density interaction that couples uniformly to every
orbital is, physically, coupling to `N_total`, and should be Lang-Firsov-eligible even
when `classify_dyn_vertices`'s per-vertex check would reject every individual piece.

**Mechanism** (verified against the actual trace code, `compute_lang_firsov_ratio` in
`qmc_data.hpp`, not just asserted): `K_n[a][b]` is a fully general matrix over
individual linear orbital indices — the trace evaluator dresses *every* individual
fermion-operator insertion with a phase from `K_n` against every other insertion, with
no assumption that `K_n` is diagonal or factorizes. If `K_n[a][b]` is set to the same
kernel for every pair `(a,b)` in some set `S` (including `a==b`), the trace math
collapses to exactly a single coupling to `N_S = sum_{a in S} n_a`. This is exact
whenever `N_S` commutes with `h_loc`, independent of whether the individual `n_a` do.

**Algorithm** (`find_total_density_decomposition` in `dynamical_interactions.cpp`):
1. Among the density-bilinear vertices, find candidates: `n_a`-`n_b` pairs (`a != b`).
2. Require *all* candidates share the exact same coupling curve (`gf_close`, numerical
   equality check) — this is what "sufficiently symmetric Hamiltonian" means in code:
   a single boson coupled uniformly to every orbital, e.g. `U(tau) == Uprime(tau)`.
3. Require completeness: every off-diagonal pair `(a,b)`, `a != b`, among the touched
   orbitals must be present (`orbitals.size() * (orbitals.size()-1)` pairs expected) —
   otherwise substituting `N_total^2` would silently add coupling for a pair the user
   never specified.
4. If both hold, build `N_total` (reusing the vertices' own operators, deduplicated by
   orbital) and remove the absorbed vertices from what `classify_dyn_vertices` sees.
5. The caller (`solver_core.cpp`) then does ONE more check before actually using this:
   does `N_total` commute with `h_loc`? (`use_total_density_decomposition` in
   `solve()`.) If not, or if the group wasn't found/complete/uniform, nothing changes
   from the existing per-vertex path — this is purely additive, never a regression.

**Diagonal correction**: populating `K_n[a][b]` uniformly over the whole group
including `a==b` reproduces `D(tau)*N_total^2`, but the *intended* physics is only
`D(tau)*sum_{a!=b} n_a n_b = D(tau)*(N_total^2 - N_total)` (using `n_a^2 = n_a`).
`apply_total_density_shift` cancels the extra `D(tau)*N_total` piece with an explicit
`+0.5*K'(0)*N_total` shift on `h_loc` (opposite sign from the usual per-orbital
`-0.5*K'(0)*n_a` shift, for exactly this reason — see the comment in the function).

**Current scope / explicit limitation**: only handles the *fully uniform* case (every
relevant pair shares the exact same coupling). Does **not** attempt a partial
decomposition when couplings differ but share a common part (e.g. Kanamori's `U` and
`U'` being different but comparable) — that's the next planned step, see below.

**Validation performed** (no exact cross-check available yet — ctseg can't do
multi-orbital, CTINT isn't set up, see below): compared `lang_firsov=True` (total-density
decomposition active, all vertices analytic) against `lang_firsov=False` (forced, same
vertices, all stochastic) on a 2-orbital full-Kanamori `h_loc` (U=3, U'=1.5, J=0.3,
spin-flip *and* pair-hopping both included) with a uniform dynamical D0 coupling. Both
give average sign exactly 1.0; `max|G_analytic - G_stochastic|` shrinks monotonically
as statistics increase (0.80/0.85 at ~20-30k measures -> 0.14/0.12 at ~150k-1.5M
measures) — consistent with converging to the same physical answer, not a systematic
bug. Validation script: `/tmp/.../scratchpad/total_density_validation.py` (in the
session scratchpad, not the repo — should be turned into a real test, see below).

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
resolved from a conflict, apparently written earlier this session) has the *wrong*
(flipped) convention and its own inline comment contradicts its own code — **not yet
fixed, listed below.**

### 8. Validated end-to-end against ctseg (real solver cross-check)

With `lang_firsov=True`, single-orbital spin-spin model, same physical inputs
(`ctint.ref.h5` bath) fed to both `triqs_ctseg` and this cthyb: both give average sign
**exactly 1.0**; `max|G_ctseg - G_cthyb|` shrinks from 3.5 (2k cycles) to 0.17 (100k
cycles) — same convergence signature as the total-density validation above. This
confirms the D0 sign convention, the Lang-Firsov pipeline, and the scalar Jperp_tau
reversion are all correct for the case ctseg can actually check (single orbital only —
ctseg cannot do multi-orbital, hence the total-density decomposition above couldn't be
cross-checked the same way).

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

**New conflict-resolution policy established this session** (saved to memory,
`feedback_merge_conflict_policy.md`): default to **unstable's side** for
multiorbital/unstable conflicts, unless there's a specific, verified reason to prefer
multiorbital's (e.g. reference/test data that provably only exists on multiorbital's
branch history, checked via `git log --oneline <branch> -- <file>` — this is why
`test/c++/spin_spin.cpp` correctly kept multiorbital's parameters, verified before
this policy was stated). If in doubt whether an already-resolved file should be
redone under this policy, ask rather than silently redo it.

Still open: whether `benchmark/dynamic_int/multiorb_spin_spin.py`'s wrong D0 sign
convention (see point 7 above) should be fixed — not a merge conflict (this file isn't
in either branch's history, appears to have been written fresh during an earlier part
of this session), just carries the same bug independently.

### 3. Partial/non-uniform total-density decomposition (next after that, most important per user)

Generalize `find_total_density_decomposition` beyond the fully-uniform case: given a
group of density vertices whose couplings are *not* all identical (e.g. Kanamori's `U`
vs `U'`), extract whatever common part they share (the user's own framing: "there
should be a way to decompose into TOTAL density and then a small residual leftover
part"), handle the common part via `N_total` (as now), and feed only the *residual*
per-pair coupling (`D_ab(tau) - D_common(tau)`) to the stochastic path — instead of
either the whole `D_ab` (current per-vertex fallback) or nothing.

Open design question, not yet resolved: how to choose `D_common(tau)` when couplings
differ in both magnitude *and* sign across pairs (e.g. Sz*Sz-type same-spin=+/opposite-spin=-
couplings, where no single common part helps and the decomposition should
correctly detect that and back off to zero decomposition). A naive pointwise-minimum
approach breaks down when signs differ; needs a well-defined, provably-correct
criterion, not a heuristic — see the earlier discussion in this session for the
mathematical reasoning about why correctness here matters more than efficiency.

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

### 5. Validation / test suite (not started)

- Turn `/tmp/.../scratchpad/total_density_validation.py` and
  `/tmp/.../scratchpad/kanamori_dyn_smoke.py`/`kanamori_full_static_test.py` into real
  tests under `test/c++` or `test/python` (currently only exist as scratchpad scripts
  from this session, will be lost otherwise).
- Regenerate `spin_spin.ref.h5` once the pure-stochastic sign problem (point 4) is
  understood — don't just regenerate blindly, since that could paper over a real bug.
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
