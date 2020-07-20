import os
from pathlib import Path
from pyomo.environ import *
from pyomo.opt.base import ProblemFormat
import xml.etree.ElementTree as ET

from problem import MolecularConformation
from .NeosClient import main as send_to_neos


class CplexNeosSolver(MolecularConformation):
    def set_hyper_parameters(self):
        cwd = Path(__file__).parents[0]
        self.LP_FILEPATH = Path.joinpath(cwd, 'model.lp')
        self.XML_TEMPLATE_FILEPATH = Path.joinpath(cwd, 'template.xml')
        self.XML_FILEPATH = Path.joinpath(cwd, 'problem.xml')

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

    def write_xml(self):
        if self.N_ATOMS * self.N_CELLS < 120:
            priority = 'short'
        else:
            priority = 'long'

        with open(self.LP_FILEPATH, 'r') as lp_file:
            xml = ET.parse(self.XML_TEMPLATE_FILEPATH)
            root = xml.getroot()
            LP = root.find('.//LP').text = lp_file.read()
            root.find('.//priority').text = priority
            xml.write(self.XML_FILEPATH)

    def cleanup(self):
        os.remove(self.LP_FILEPATH)
        os.remove(self.XML_FILEPATH)

    def solve(self):
        print('creating model...', end='')
        self.write_model()
        print(' DONE')
        print('creating XML...', end='')
        self.write_xml()
        print(' DONE')
        print('sending to NEOS server...', end='')
        send_to_neos(self.XML_FILEPATH)
        print(' DONE')
        print('cleaning up...', end='')
        self.cleanup()
        print(' COMPLETE')
