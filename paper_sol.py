'''
The solution to the 4 atom butane problem in a 5x5x5 lattice
as given by Phillips 1994.
For comparison to gotten results
'''

import numpy as np
from dwavesolver import DwaveSolver
from collections import defaultdict

s = DwaveSolver(4, 5)
solution = defaultdict(int)
solution[90] = 1
solution[196] = 1
solution[317] = 1
solution[437] = 1

# i,j
# X[(0, 90)] = 1
# X[(1, 71)] = 1
# X[(2, 67)] = 1
# X[(3, 62)] = 1

positions = s.sample_to_positions(solution)
# positions = np.array([[0, 3.0, 3], [1, 4, 2], [2, 3, 2], [2, 2, 2]])

print("Energy:", s.objective_value(s.sample_to_x_ij_matrix(solution).flat))
# s.plot_3d(positions)
