import os
from pathlib import Path
from pyomo.environ import *
from pyomo.opt.base import ProblemFormat
import xml.etree.ElementTree as ET

from lpwriter import LpWriter
from .NeosClient import main as send_to_neos


class CplexNeosSolver(LpWriter):
    def set_hyper_parameters(self):
        cwd = Path(__file__).parents[0]
        self.LP_FILEPATH = Path.joinpath(cwd, 'model.lp')
        self.XML_TEMPLATE_FILEPATH = Path.joinpath(cwd, 'template.xml')
        self.XML_FILEPATH = Path.joinpath(cwd, 'problem.xml')

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
        print('sending to NEOS server...')
        print('(It should be safe to force quit once "BEGIN SOLVER OUTPUT" appears)')
        send_to_neos(self.XML_FILEPATH)
        print(' DONE')
        print('cleaning up...', end='')
        self.cleanup()
        print(' COMPLETE')
