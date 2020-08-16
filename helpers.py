import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class EquationsMixin:
    def leonard_jones_potential(self, distance):
        """
        eq (1) in paper, error in paper? Following cited paper.
        Possibly lennard jones.
        """
        # ignore divide by 0 and nan subtraction
        with np.errstate(divide='ignore', invalid='ignore'):
            div = (self.SIGMA / distance)
            div6 = div ** 6
            return 4 * self.e * (div6 ** 6 - 2 * div6)

    def bond_stretching_potential(self, distance, is_next):
        """
        eq (2) in paper, only applies if the atoms are neighbors
        is_next should be 0-1 matrix
        """
        return is_next * self.BETA * (distance - self.bond_length) ** 2

    def total_potential(self, distance, is_next):
        """
        total potential in terms of distance
        """
        lj = self.leonard_jones_potential(distance)
        bond = self.bond_stretching_potential(distance, is_next)
        return np.nan_to_num(lj + bond, 0)

    def potential_from_indices(self, i, j, k, l):
        # j and l are the spot index, change to coordinates
        j_loc = self.index_to_location(j)
        l_loc = self.index_to_location(l)

        # distance between j and l
        dists = self.distance(j_loc, l_loc)

        k_is_i_plus_1 = 1 * (k == i + 1)

        return self.total_potential(dists, k_is_i_plus_1)

    def objective_value(self, solution):
        total = 0
        ones = filter(lambda t: t[1] == 1, enumerate(solution))
        for (ij, one) in ones:
            [i, j] = self.q_to_ij(ij)
            for (kl, one) in ones:
                [k, l] = self.q_to_ij(kl)
                total += self.potential_from_indices(i, j, k, l)
        return total


class UtilsMixin:
    def distance(self, x1, x2):
        """
        Computes the euclidean distance between points in R^3
        """
        return np.sum((self.CELL_LENGTH * (x1 - x2)) ** 2, 0) ** 0.5

    def ij_to_q(self, i, j):
        """
        Returns the corresponding index in the hamiltonian of the binary variable
        corresponding to atom i being in spot j
        """
        # flattens matrix of binary vars to array
        return i * self.N_CELLS + j

    def q_to_ij(self, q_index):
        """
        Given a binary variable index returns i, j corresponding to atom i being in spot j
        """
        return (np.floor(q_index / self.N_CELLS), q_index % self.N_CELLS)

    def index_to_location(self, i):
        """
        Returns the euclidean coordinates for spot i
        """
        div = np.array([i / self.LATTICE_LENGTH ** d for d in range(3)])
        mod = np.floor(div) % self.LATTICE_LENGTH
        return mod

    def is_solution_valid(self, X):
        """
        Input: np solution matrix
        Returns a boolean if the solution fulfills basic
        problem constraints
        """
        every_atom_in_one_spot = (np.sum(X, 1) == 1).all()
        spot_has_zero_or_one_atoms = (np.sum(X, 0) <= 1).all()
        correct_number_of_atoms = np.sum(X) == self.N_ATOMS
        return every_atom_in_one_spot and spot_has_zero_or_one_atoms and correct_number_of_atoms

    def sample_to_x_ij_matrix(self, sample):
        """
        Turns a sample from the form of a dict {(i,j): 0|1, ...} into a np
        0-1 matrix of size N_ATOMS x N_CELLS  
        """
        X = np.zeros([self.N_ATOMS, self.N_CELLS])
        ones = filter(lambda i: i[1] == 1, dict(sample).items())
        for qi, _ in ones:
            (i, j) = self.q_to_ij(qi)
            X[math.floor(i), j] = 1
        return X

    def sample_to_positions(self, sample):
        """
        Turns a sample from the form of a dict {(i,j): 0|1, ...} into a np
        matrix of size N_ATOMS x 3 which is a list of the grid positions in R^3
        """
        positions = np.zeros([self.N_ATOMS, 3])
        ones = filter(lambda i: i[1] == 1, dict(sample).items())
        for qi, _ in ones:
            (i, j) = self.q_to_ij(qi)
            positions[math.floor(i), :] = self.index_to_location(j)
        return positions

    def plot_3d(self, positions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1],
                   positions[:, 2], s=80, c='r')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        plt.show()
