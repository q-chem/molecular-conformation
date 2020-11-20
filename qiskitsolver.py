from dotenv import load_dotenv
import os
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import GroverOptimizer
from qiskit import BasicAer

from lpwriter import LpWriter

load_dotenv()


class QiskitSolver(LpWriter):
    def solve(self):
        # self.write_model()
        # IBMQ.enable_account(os.getenv('IBM_TOKEN'))
        backend = BasicAer.get_backend('ibmq_qasm_simulator')
        # backend = Aer.get_backend('qasm_simulator')
        prog = QuadraticProgram("molecConform")

        prog.read_from_lp_file(str(self.LP_FILEPATH))
        optimizer = GroverOptimizer(3, quantum_instance=backend)
        results = optimizer.solve(prog)
        print(results)
        import pdb
        pdb.set_trace()
