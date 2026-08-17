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
#SBATCH --time=72:00:00
MODULES="devenv9/clang-py3-mkl llvm/20"
module purge
module use /mnt/home/wentzell/opt/modules
module load ${MODULES}
module load triqs/unstable
##################### run your code here #####################

# Dynamical Hubbard-Kanamori benchmark: full static Kanamori (U, U'=U-2J, J, spin-flip
# + pair-hopping) with a phonon coupled uniformly to total density, specified completely
# (off-diagonal + diagonal self-terms) so it's exactly recoverable to the analytic
# Lang-Firsov path via recover_conserved_density_groups -- see kanamori_dyn.py.
mpirun -n 96 python kanamori_dyn.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.5 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov True

# Same physical setup, forced pure-stochastic instead (for comparison against the run above):
#mpirun -n 96 python kanamori_dyn.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.5 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov False

# Lang-Firsov vs. forced-stochastic self-consistency check in one job (saves both to one file):
#mpirun -n 96 python kanamori_dyn_selfconsistency.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.5 --omega_0 1.0 --beta 10.0 --n_cycles_lf 1000000 --n_cycles_stoch 2000000
