import neal
from problem import MolecularConformation
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
from cliparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    p = MolecularConformation(args.num_molecules, args.lattice_size)

    if args.solver == 'hybrid':
        solver = LeapHybridSampler()
    elif args.solver == 'embed':
        EmbeddingComposite(DWaveSampler())
    elif args.solver == 'sim_anneal':
        neal.SimulatedAnnealingSampler()
    else:
        # args.solver == 'tabu'
        solver = 'tabu'

    p.solve(
        solver=solver,
        visualize=args.visualize,
        top_samples=args.sols_to_print
    )
