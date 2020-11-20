import neal
from dwavesolver import DwaveSolver
from qiskitsolver import QiskitSolver
from cplexsolver import CplexNeosSolver
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
from cliparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    options = {}
    if args.solver in ['hybrid', 'embed', 'sim_anneal', 'tabu']:
        solverClass = DwaveSolver

        if args.solver == 'hybrid':
            options['solver'] = LeapHybridSampler()
        elif args.solver == 'embed':
            options['solver'] = EmbeddingComposite(DWaveSampler())
        elif args.solver == 'sim_anneal':
            options['solver'] = neal.SimulatedAnnealingSampler()
        elif args.solver == 'tabu':
            options['solver'] = 'tabu'

        options['visualize'] = args.visualize
        options['top_samples'] = args.sols_to_print

    elif args.solver == 'cplex':
        solverClass = CplexNeosSolver
    elif args.solver == 'qiskit':
        solverClass = QiskitSolver

    solver = solverClass(args.num_molecules, args.lattice_size)
    solver.solve(options)
