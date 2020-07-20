import neal
from dwavesolver import DwaveSolver
from cplex.solver import CplexNeosSolver
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
from cliparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    if args.solver in ['hybrid', 'embed', 'sim_anneal', 'tabu']:
        controller = DwaveSolver(args.num_molecules, args.lattice_size)

        if args.solver == 'hybrid':
            solver = LeapHybridSampler()
        elif args.solver == 'embed':
            solver = EmbeddingComposite(DWaveSampler())
        elif args.solver == 'sim_anneal':
            solver = neal.SimulatedAnnealingSampler()
        else:
            # args.solver == 'tabu'
            solver = 'tabu'

        controller.solve(
            solver=solver,
            visualize=args.visualize,
            top_samples=args.sols_to_print
        )

    elif args.solver == 'cplex':
        controller = CplexNeosSolver(args.num_molecules, args.lattice_size)
        controller.solve()
