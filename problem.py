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
        self.BETA = 1000000             # scalar for bond stretching penalty

    def set_problem_parameters(self, N_ATOMS, LATTICE_LENGTH):
        self.N_ATOMS = N_ATOMS         # B in paper
        self.LATTICE_LENGTH = LATTICE_LENGTH
        self.N_CELLS = self.LATTICE_LENGTH ** 3        # 4x4x4 = 64, N in paper

    # abstract class for solver spec
    def set_hyper_parameters(self):
        """
        Abstract method for solver specific parameters
        """
        pass
