from triqs.gfs import *
import numpy as np

print("Testing Boson Legendre Mesh with 10 points")
g10 = GfLegendre(indices=[0], beta=10, statistic='Boson', n_points=10)
print(g10)

print("Testing Boson Legendre Mesh with 100 points")
g100 = GfLegendre(indices=[0], beta=10, statistic='Boson', n_points=100)
print(g100)
