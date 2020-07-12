import numpy as np


class HelperMethodsMixin:
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

    def objective_value(self, solution):
        total = 0
        for (ij, x_ij) in enumerate(solution):
            [i, j] = self.x_index(ij)
            for (kl, x_kl) in enumerate(solution):
                [k, l] = self.x_index(kl)
                is_next = i + 1 == k
                dist = self.distance(
                    self.index_to_location(j),
                    self.index_to_location(l)
                )
                total += self.total_potential(dist, is_next)
        return total
