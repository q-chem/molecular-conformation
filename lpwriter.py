import os
from pathlib import Path
from pyomo.environ import *
from pyomo.opt.base import ProblemFormat

from problem import MolecularConformation


class LpWriter(MolecularConformation):
    def set_hyper_parameters(self):
        cwd = Path(__file__).parents[0]
        self.LP_FILEPATH = Path.joinpath(cwd, 'model.lp')

    def write_model(self):
        model = ConcreteModel()

        model.B = self.N_ATOMS
        model.N = self.N_CELLS

        model.I = RangeSet(0, model.B - 1)
        model.J = RangeSet(0, model.N - 1)
        model.K = RangeSet(0, model.B - 1)
        model.L = RangeSet(0, model.N - 1)

        model.x = Var(model.I, model.J, within=Binary)

        def actual_potential(model, i, j, k, l):
            return model.x[i, j] * model.x[k, l] * self.potential_from_indices(i, j, k, l)

        def objective_func(model):
            return sum(
                actual_potential(model, i, j, k, l)
                for i in model.I for j in model.J
                for k in model.K for l in model.L
            )

        def every_atom_in_one_place(model, i):
            return sum(model.x[i, j] for j in model.J) == 1

        model.InOneSpotCon = Constraint(model.I, rule=every_atom_in_one_place)

        def no_more_than_one_atom_per_space(model, j):
            return sum(model.x[i, j] for i in model.I) <= 1

        model.AtMostOneAtomCon = Constraint(
            model.J, rule=no_more_than_one_atom_per_space)

        model.OBJ = Objective(rule=objective_func)

        res = model.write(self.LP_FILEPATH, format=ProblemFormat.cpxlp)

    def cleanup(self):
        os.remove(self.LP_FILEPATH)
