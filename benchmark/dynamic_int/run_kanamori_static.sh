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
module load triqs/multiorbital
##################### run your code here #####################

# Static-only reference for the kanamori_dyn.py comparison: same Kanamori U, U'=U-2J,
# J, spin-flip + pair-hopping, and the same mu_half formula and h_loc0 sign, but with
# g=0 so there's no phonon coupling (kanamori_dynamical_vertices/add_dyn_vertex still
# run, but Q_tau is identically zero so 0 vertices are added -- exactly the code path
# for the dynamical run below, minus the dynamical interaction). Compare the resulting
# density and G_tau against run_kanamori_dyn.sh's g=0.5 run to check that both sit at
# half filling and to see what the retarded interaction actually changes.
mpirun -n 96 python kanamori_dyn.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.0 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov True
