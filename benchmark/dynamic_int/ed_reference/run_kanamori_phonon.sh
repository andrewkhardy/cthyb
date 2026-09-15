#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
module load modules/2.5-beta1
module load triqs/multiorbital
##################### run your code here #####################

# CTHYB against an exact-diagonalization reference for the same Hamiltonian: 2-orbital
# Kanamori (spin-flip + pair-hopping) + one bath site per spin-orbital + one phonon whose
# coupling g_a may differ per orbital. Unequal g is the "U_tot + dU" case: split_density_couplings
# sends the block-constant part to Lang-Firsov and the residual to the stochastic expansion, and
# the residual vertices are what the coupling-derivative estimator (dyn_vertex_corr_tau) measures.
# See model.py for the Hamiltonian and ed_kanamori_phonon.py for the reference.
#
# These runs are short; the allocation above is generous on purpose.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
DATA=/mnt/home/ahardy/ceph/CTHYB_Data/ed_reference
mkdir -p ${DATA}

# MODEL goes to both scripts, so the two solve the same Hamiltonian; MC is CTHYB-only
# (ed_kanamori_phonon.py takes no Monte Carlo arguments).
MODEL="--beta 10.0 --U 2.0 --J 0.3 --V 0.7 --omega_0 1.0 --out_dir ${DATA}"
MC="--n_cycles 1000000 --dyn_n_l 50"

# The exact reference, serial and cheap (seconds). Must use the SAME model parameters as the
# CTHYB run below, or plot_ed_vs_cthyb.py will refuse to compare them. It prints its own
# self-tests: phonon truncation, <n_a> = 0.5, and G(0) + G(beta) = -1, all at ~1e-15.
srun -n 1 python ed_kanamori_phonon.py ${MODEL} --g 0.7 0.3

# Orbital-dependent coupling: expect "16 Lang-Firsov and 12 stochastic residual vertex(es)"
# and average sign 1.0. The 12 residual vertices cover the orbital-0 and mixed pairs; the
# orbital-1 residual vanishes by construction, so no correlator is measured for that pair.
mpirun -n 96 python cthyb_kanamori_phonon.py ${MODEL} ${MC} --g 0.7 0.3 --lang_firsov True

# Uniform coupling: everything is block-constant, so it all goes analytic (16 Lang-Firsov,
# 0 stochastic) and there is nothing for the vertex estimator to measure. Control for the
# Lang-Firsov path itself against ED.
#srun -n 1 python ed_kanamori_phonon.py ${MODEL} --g 0.5 0.5
#mpirun -n 96 python cthyb_kanamori_phonon.py ${MODEL} ${MC} --g 0.5 0.5 --lang_firsov True

# Phonon on orbital 0 only: the case the old recover_conserved_density_groups got wrong (it
# treated the non-conserved n_0 as conserved). The blocks now contain uncoupled pairs, so the
# sign-preserving split takes nothing analytically and all 4 vertices stay stochastic.
#srun -n 1 python ed_kanamori_phonon.py ${MODEL} --g 0.5 0.0
#mpirun -n 96 python cthyb_kanamori_phonon.py ${MODEL} ${MC} --g 0.5 0.0 --lang_firsov True

# Same physics forced fully stochastic, as an independent route to the same answer: every
# vertex goes through insert_dyn/remove_dyn, so the estimator covers every pair, at the cost
# of a higher expansion order.
#mpirun -n 96 python cthyb_kanamori_phonon.py ${MODEL} ${MC} --g 0.7 0.3 --lang_firsov False

# If the run dies in post-processing with "Can not wrap AtomDiagReal ... legacy cpp2py", the
# local atom_diag/c2py mismatch is present in this environment: rerun with --density_matrix False.
# Q_conserved_tau is then missing its equal-time constant, but dyn_vertex_corr_tau is unaffected
# (the coupling-derivative estimator carries its own equal-time value).

##### After the run, inspect with:
# python plot_ed_vs_cthyb.py ${DATA}/ed_beta-10.0_U-2.0_J-0.3_V-0.7_eb-0.0_w0-1.0_g-0.7-0.3_mu-half_nph-24.h5 \
#                            ${DATA}/cthyb_beta-10.0_U-2.0_J-0.3_V-0.7_eb-0.0_w0-1.0_g-0.7-0.3_mu-half_lf-True_nc-1000000.h5
# Two figures: *_vs_ed.png (G and the conserved-combination correlators) and *_vs_ed_chi_ab.png
# (orbital-resolved <n_a(tau) n_b(0)> from the stochastic vertices, against ED).
