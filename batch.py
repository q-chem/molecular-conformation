from dwavesolver import DwaveSolver

# SUB_QUBO_SIZES = [12, 24, 48, 64, 100, 128, 200]
SUB_QUBO_SIZES = [12, 24]
PROB_SIZES = [
    [4, 3],
    # [5, 5],
    # [8, 6],
    # [8, 8]
]

if __name__ == "__main__":
    solverClass = DwaveSolver
    options = {
        'verbosity': 0,
        'solver': 'tabu',
        'solver_limit': 48,
        'no_time': False,
        'top_samples': 0
    }
    for [B, L] in PROB_SIZES:
        solver = solverClass(B, L)
        for sub_size in SUB_QUBO_SIZES:
            options['solver_limit'] = sub_size
            print("---------------------------")
            print(f"ATOMS {B}, LATTICE {L}, sub_size {sub_size}")
            solver.solve(options)
