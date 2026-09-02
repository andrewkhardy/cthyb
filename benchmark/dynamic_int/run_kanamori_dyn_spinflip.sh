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

# Dynamical Hubbard-Kanamori benchmark with a genuine dynamical (retarded) spin-flip
# channel on top of the usual density-density phonon coupling -- see
# kanamori_dyn_spinflip.py's module docstring for the physics/classification detail.
# Unlike kanamori_dyn.py's fully-recovered runs, this always has nonzero stochastic
# vertices (the spin-flip channel), even with lang_firsov=True: it's the first
# production run exercising the stochastic double expansion for a genuinely
# multi-orbital dynamical Jperp-type vertex built via kanamori_dynamical_vertices,
# rather than the single-orbital scalar Jperp_tau path (spin_spin.cpp/.py).
mpirun -n 96 python kanamori_dyn_spinflip.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.5 --g_sf 0.5 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov True

# Same physical setup, forced fully stochastic instead (density part included, for
# comparison against the run above):
#mpirun -n 96 python kanamori_dyn_spinflip.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.5 --g_sf 0.5 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov False

# Stronger spin-flip coupling -- better statistics on the stochastic channel at some
# cost in sign (short local scan at these parameters: g_sf=0.5 -> sign 0.89, ~7% of
# measures carry a spin-flip vertex; g_sf=0.7 -> 0.86/~11%; g_sf=1.0 -> 0.75/~26%):
#mpirun -n 96 python kanamori_dyn_spinflip.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.5 --g_sf 0.7 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov True

# Spin-flip channel only (g=0: the density vertices are still registered and still go
# analytic, but with an identically-zero coupling, so the only physical dynamical
# interaction left is the stochastic spin-flip one; mu correction is then a no-op):
#mpirun -n 96 python kanamori_dyn_spinflip.py --n_orb 2 --U 2.0 --Jhund 0.2 --g 0.0 --g_sf 0.5 --omega_0 1.0 --beta 10.0 --n_cycles 1000000 --dyn_n_l 100 --lang_firsov True

##### After the run, inspect with:
# python plot_kanamori_dyn_spinflip.py /mnt/home/ahardy/ceph/CTHYB_Data/kanamori_dyn_spinflip_cthyb_*.h5 --out spinflip_comparison.png
