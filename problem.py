import dimod
from dwave_qbsolv import QBSolv
import numpy as np
from helpers import EquationsMixin, UtilsMixin


class MolecularConformation(EquationsMixin, UtilsMixin):
    def __init__(self, N_ATOMS, LATTICE_LENGTH):
        self.set_parameters(N_ATOMS, LATTICE_LENGTH)

    def set_parameters(self, N_ATOMS, LATTICE_LENGTH):
        self.set_problem_parameters(N_ATOMS, LATTICE_LENGTH)
        self.set_hyper_parameters()

        # physical constants
        self.CELL_LENGTH = 1.4          # A
        self.SIGMA = 3.6                # A
        self.e = 0.06                   # kcal
        self.bond_length = 1.5          # A approximate

    def set_problem_parameters(self, N_ATOMS, LATTICE_LENGTH):
        self.N_ATOMS = N_ATOMS         # B in paper
        self.LATTICE_LENGTH = LATTICE_LENGTH
        self.N_CELLS = self.LATTICE_LENGTH ** 3        # 4x4x4 = 64, N in paper

    def set_hyper_parameters(self):
        self.A = 1                      # scalar for constraint matrix
        self.B = 12000                  # scalar for objective matrix
        self.BETA = 1000000             # scalar for bond stretching penalty

    def constraint_Q(self, scale=False):
        """
        Enforces that every spot has at most 1 atom and that
        every atom is only in 1 spot.
        Matrix has a regular structure.
        returns an upper triangular np matrix that represents the last 2
        terms in the hamiltonian in (eq 8)
        """
        N = self.N_ATOMS * self.N_CELLS
        Q = np.zeros([N, N])
        Q1 = 2 * np.triu(np.ones([self.N_CELLS, self.N_CELLS]))
        np.fill_diagonal(Q1, -1)
        Q2 = np.zeros([self.N_CELLS, self.N_CELLS])
        np.fill_diagonal(Q2, 2)

        line = np.concatenate([Q1] + [Q2] * (self.N_ATOMS - 1), axis=1)
        for i in range(self.N_ATOMS):
            start_row = i * self.N_CELLS
            Q[start_row:start_row + self.N_CELLS, :] = np.roll(line, start_row)

        Q = np.triu(Q)
        if scale:
            return Q / Q.max()
        return Q

    def objective_Q(self, scale=False):
        """
        First term in hamiltonian in eq 8.
        Returns an np matrix
        """
        N = self.N_ATOMS * self.N_CELLS
        [x, y] = np.indices((N, N))     # x,y indices on the hamiltonian

        # to binary var x_ij and x_kl
        [i, j] = self.q_to_ij(x)
        [k, l] = self.q_to_ij(y)

        # j and l are the spot index, change to coordinates
        j_loc = self.index_to_location(j)
        l_loc = self.index_to_location(l)

        # distance between j and l
        dists = self.distance(j_loc, l_loc)

        k_is_i_plus_1 = 1 * (k == i + 1)

        potentials_full = self.total_potential(dists, k_is_i_plus_1)
        potentials = 2 * np.triu(potentials_full)

        if scale:
            return potentials / potentials.max()
        return potentials

    def make_Q(self):
        """
        Returns the complete hamiltonian as a np matrix
        """
        return self.A * self.constraint_Q(scale=True) + self.B * self.objective_Q(scale=True)

    def solve(self, solver='tabu', top_samples=1, visualize=False):
        # get hamiltonian
        q = self.make_Q()

        # from np matrix to dwave representation
        Q = dimod.BinaryQuadraticModel.from_numpy_matrix(q)

        # solve using QBSolv
        response = QBSolv().sample(Q, verbosity=0, solver=solver)

        for sample_i, sample in enumerate(response.samples()[0:top_samples]):
            X = self.sample_to_x_ij_matrix(sample)
            print(f'------- sample {sample_i} -------')
            print('solution is valid:', self.is_solution_valid(X))
            print('energy:', response.record.energy[sample_i])
            if visualize:
                positions = self.sample_to_positions(sample)
                self.plot_3d(positions)
