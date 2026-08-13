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
#module load triqs/multiorbital
module load triqs/unstable
##################### run your code here #####################
#../CTHYB/benchmark/
# Alternate environment activation, if triqs/unstable module isn't what's wanted:
#source /mnt/home/ahardy/ccq-software-build/triqs/.triqs_dev/bin/activate
#source /mnt/home/ahardy/ccq-software-build/triqs/3_development/installation/share/triqs/triqsvars.sh

#mpirun -n 96 python spin_spin.py --J 0.5 --U 4.0 --i1 1.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0 --n_cycles 1000000 --measure_O_tau 22 --dyn_n_l 22
mpirun -n 96 python ctseg_spin_spin.py --J 0.5 --U 4.0  --i1 1.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0
#mpirun python multiorb_spin_spin.py --J 1.0 --U 4.0 --i1 1.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0 --n_cycles 10000000 --measure_O_tau 10
