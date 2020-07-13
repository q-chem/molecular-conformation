import argparse

parser = argparse.ArgumentParser(
    description="""Try to find a good solution for a molecular conformation
        problem using d-wave utilities and a variety of solvers."""
)

parser.add_argument(
    '-B', '--num_molecules',
    type=int,
    default=4,
    help='how many molecules the problem should have. Default 4'
)
parser.add_argument(
    '-L', '--lattice_size',
    type=int,
    default=4,
    help='how large the lattice is (one side, so L=4 means 4**3 = 64 spots). Default 4'
)
parser.add_argument(
    '-s', '--solver',
    type=str,
    default='tabu',
    choices=['tabu', 'hybrid', 'embed', 'sim_anneal'],
    help='which solver to use. Default "tabu"'
)
parser.add_argument(
    '-v', '--visualize',
    dest='visualize',
    action='store_true',
    help='optional flag, if included the results will be visualized on a 3D grid',
    required=False
)
parser.add_argument(
    '-p', '--sols_to_print',
    type=int,
    help='How many of the top solutions to print out.',
    default=1
)
