# Shared helpers for the dynamic_int benchmarks: retarded interaction kernels at any beta
# (kernels), one self-energy convention for every solver (selfenergy), one tau and Matsubara
# grid for every solver and ED (grids), and one h5 schema (io). Each benchmark's model.py
# puts this directory's parent on sys.path and imports
# `from common import ...`.
