#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com 
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=ccq
#SBATCH --constraint=icelake
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00

module purge
#source /mnt/home/ahardy/ccq-software-build/triqs/3_unst_nix2.3_llvm/installation/share/triqs/triqsvars.sh
#source /mnt/home/ahardy/ccq-software-build/triqs/.triqs_dev/bin/activate
source /mnt/home/ahardy/ccq-software-build/triqs/unstable/installation/share/triqs/triqsenv.sh
#source /mnt/home/ahardy/ccq-software-build/triqs/3_nu_nu/installation/share/triqs/triqsvars.sh

##################### run your code here #####################
#../CTHYB/benchmark/
srun python compare_ctint_ctseg.py 
#mpirun python spin_spin.py --J 0.25 --U 4.0 --i1 1.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0 --n_cycles 1000000 --measure_O_tau 50
#mpirun python ctseg_spin_spin.py --J 1.0 --U 4.0 --i1 0.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
#mpirun python ctseg_spin_spin.py --J 1.0 --U 4.0 --i1 0.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0
#mpirun python ctseg_spin_spin.py --J 1.0 --U 4.0 --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0