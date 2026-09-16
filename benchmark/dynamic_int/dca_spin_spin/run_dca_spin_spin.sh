#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
module load modules/2.5-beta1
module load triqs/multiorbital
##################### run your code here #####################

# Two-patch DCA with a *real-space* retarded full S.S interaction. Two orbitals = the two
# patch-averaged Fermi-surface points of the 2D Hubbard model (the VBDMFT patches,
# |kx|,|ky| < pi/sqrt(2) and the rest). The static Hubbard U and the retarded S.S are both
# local on the two *rotated* cluster sites, so in the solver's working (patch) basis they
# are off-diagonal: the dynamical vertices are bilinears c^dag_{K s} c_{K' s'} with K != K'.
# See dca_model.py for the Hamiltonian and the monomial expansion that makes this
# expressible through add_dyn_vertex (which only accepts one bilinear per operator).
#
# This is the first benchmark here whose dynamical interaction is off-diagonal in the
# working basis, and the first where the plain per-vertex Lang-Firsov test rejects
# everything (n_{K sigma} does not commute with h_loc), leaving only the span{N_up, N_down}
# projector split.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
DATA=/mnt/home/ahardy/ceph/CTHYB_Data/dca_spin_spin
mkdir -p ${DATA}

MODEL="--beta 10.0 --t 0.25 --tp 0.0 --U 2.0 --J_intra 0.0 --J_inter 0.5 --omega_0 1.0 --out_dir ${DATA}"
MC="--n_cycles 1000000 --n_warmup_cycles 50000 --dyn_n_l 50"

##### 0. Free checks -- run these before spending anything.
# Symbolic only, about a second, no solver. Pins the sign conventions to spin_spin.py,
# proves the rotation and the monomial expansion are exact (via the basis independence of
# S_tot.S_tot), and reports how many vertices the run will carry. Expect "All checks
# passed" and, for J_intra 0 / J_inter 0.5, "192 raw monomial pairs -> 48 vertices".
srun -n 1 python check_rotation.py ${MODEL}

# Builds the model, the coarse-grained hybridization and the whole vertex list, prints
# them and stops before any Monte Carlo. Check: patch levels ~ -+0.359, Delta off-diagonal
# exactly 0, 48 vertices with |coefficient| <= 0.125.
srun -n 1 python cthyb_dca_spin_spin.py ${MODEL} ${MC} --dry_run True

##### 1. The two production runs. Same physics, two independent routes -- this pair IS the
##### benchmark until the ED reference exists.
# lang_firsov True: the part of the density coupling lying in span{N_up, N_down} is resummed
# analytically, the rest is sampled. Expect the solver to report a split of the 16
# density-density vertices, with the 32 off-diagonal ones always stochastic.
mpirun -n 96 python cthyb_dca_spin_spin.py ${MODEL} ${MC} --lang_firsov True

# lang_firsov False: every vertex goes through insert_dyn/remove_dyn/swap_dyn. Higher
# expansion order and a worse sign, but it shares no code with the analytic path, and it is
# the run whose coupling-derivative estimator covers *all* 48 vertices -- so it is the one
# the site-resolved <S_i(tau).S_j(0)> reconstruction needs.
mpirun -n 96 python cthyb_dca_spin_spin.py ${MODEL} ${MC} --lang_firsov False

##### 2. Controls, if the pair above disagrees or the sign is bad.
# Uniform J: sum_{ij} S_i.S_j collapses to S_tot.S_tot, which is basis independent, so the
# rotation drops out entirely and the vertex list shrinks from 48 to 24. If this agrees
# with a patch-basis calculation but the non-uniform case does not, the problem is in the
# rotated sector specifically, not in the S.S machinery.
#mpirun -n 96 python cthyb_dca_spin_spin.py ${MODEL/--J_intra 0.0/--J_intra 0.5} ${MC} --lang_firsov True

# J = 0: pure static two-patch DCA. Must reproduce a plain U-only run (and vbdmft.py's
# structure at the same parameters), confirming the DCA scaffolding itself is right.
#mpirun -n 96 python cthyb_dca_spin_spin.py ${MODEL/--J_inter 0.5/--J_inter 0.0} ${MC} --lang_firsov True

# Discrete bath instead of the coarse-grained one: one bath site per patch and spin, so the
# problem is exactly representable in exact diagonalization. Not the DCA bath -- this is the
# variant the ED reference will target.
#mpirun -n 96 python cthyb_dca_spin_spin.py ${MODEL} ${MC} --bath discrete --V 0.5 --lang_firsov False

# Seed sweep, to tell a bias from noise (see diagnose_residual.py in ed_reference/ for why
# the l = 0 channel is the one to watch). The solver's default seed is fixed, so a plain
# rerun is bit-identical: --random_seed is the only way to get independent samples.
#for SEED in 1 2 3 4 5 6; do
#  mpirun -n 96 python cthyb_dca_spin_spin.py ${MODEL} ${MC} --lang_firsov False --random_seed ${SEED}
#done

# If a run dies in post-processing with "Can not wrap AtomDiagReal ... legacy cpp2py", that
# is the local atom_diag/c2py mismatch: --density_matrix is already False by default here.

##### After the run: pull the data back (a few MB), then analyse.
# FIsync CTHYB_Data/dca_spin_spin /home/andrewhardy/Documents/Data/CTHYB_Data
#
# Edit DATA/LF_TRUE at the top of plot_dca_spin_spin.py, then run it (or paste it into a
# notebook). It prints the lang_firsov True-vs-False deviation in G_tau and in
# <S_tot^z S_tot^z> -- the cross-check -- and plots the site-resolved <S_i(tau).S_j(0)>
# rebuilt from the per-vertex correlators.
