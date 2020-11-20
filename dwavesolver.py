import dimod
import numpy as np
from dwave_qbsolv import QBSolv

from problem import MolecularConformation


class DwaveSolver(MolecularConformation):
    def set_hyper_parameters(self):
        self.A = 1000000                      # scalar for constraint matrix
        self.B = 12000                  # scalar for objective matrix

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

        potentials_full = self.potential_from_indices(i, j, k, l)
        potentials = 2 * np.triu(potentials_full)

        if scale:
            return potentials / potentials.max()
        return potentials

    def make_Q(self):
        """
        Returns the complete hamiltonian as a np matrix
        """
        return self.A * self.constraint_Q(scale=True) + self.B * self.objective_Q(scale=True)

    # def solve(self, solver='tabu', top_samples=1, visualize=False):

    def solve(self, options):
        """
        Options:
            'solver' - see solver input to Qbsolve.sample, default 'tabu'
            'top_samples' - int, how many samples to print
            'visualize' - boolean,
            'verbosity' - int, default 0 (low)
        """
        DEFAULT_OPTIONS = {
            'solver': 'tabu',
            'top_samples': 1,
            'visualize': False,
            'verbosity': 0
        }
        complete_options = DEFAULT_OPTIONS.copy()
        complete_options.update(options)

        # get hamiltonian
        q = self.make_Q()

        # from np matrix to dwave representation
        Q = dimod.BinaryQuadraticModel.from_numpy_matrix(q)

        # solve using QBSolv
        response = QBSolv().sample(
            Q,
            verbosity=complete_options['verbosity'],
            solver=complete_options['solver']
        )

        for sample_i, sample in enumerate(response.samples()[0:complete_options['top_samples']]):
            X = self.sample_to_x_ij_matrix(sample)
            print(f'------- sample {sample_i} -------')
            print('solution is valid:', self.is_solution_valid(X))
            print('energy:', response.record.energy[sample_i])
            print('total U:', self.objective_value(X.flat))

            if complete_options['visualize']:
                positions = self.sample_to_positions(sample)
                self.plot_3d(positions)
