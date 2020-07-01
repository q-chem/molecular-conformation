
from collections import defaultdict
import dimod
from utils import print_Q, visualize
import numpy as np
import math


def leonard_jones_potential(distance):
    SIGMA = 3.6     # A
    e = 0.06        # kcal
    return 4 * e * ((SIGMA / distance) ** 12 - (SIGMA / distance) ** 6)


def bond_stretching_potential(beta, distance):
    bond_length = 1.5       # A approximate
    return beta * (distance - bond_length) ** 2


def total_potential(beta, distance):
    return leonard_jones_potential(distance) + bond_stretching_potential(beta, distance)


def distance(x1, x2):
    return np.sum((x1 - x2) ** 2, 0) ** 0.5


N_ATOMS = 3         # B in paper
LATTICE_LENGTH = 2
N_CELLS = LATTICE_LENGTH ** 3        # 4x4x4 = 64, N in paper
CELL_LENGTH = 1.4   # A


def Q_index(i, j):
    # flattens matrix of binary vars to array
    return i * N_CELLS + j


def x_index(q_index):
    return (np.floor(q_index / N_CELLS), q_index % N_CELLS)


def index_to_location(i):
    div = np.array([i / LATTICE_LENGTH ** d for d in range(3)])
    mod = np.floor(div) % LATTICE_LENGTH
    return mod


def make_Q():
    Q = defaultdict(int)
    A = 1
    for i in range(N_ATOMS):
        for j in range(N_CELLS):
            q_ij = Q_index(i, j)
            Q[(q_ij, q_ij)] -= 1
            for k in range(j+1, N_CELLS):       # ?
                q_ik = Q_index(i, k)
                Q[(q_ij, q_ik)] += 2

    for j in range(N_CELLS):
        for i in range(N_ATOMS):
            q_ij = Q_index(i, j)
            # Q[(q_ij, q_ij)] -= 1
            for k in range(i + 1, N_ATOMS):
                q_kj = Q_index(k, j)
                Q[(q_ij, q_kj)] += 2
    return Q


def constraint_Q():
    A = 1
    N = N_ATOMS * N_CELLS
    Q = np.zeros([N, N])
    Q1 = 2 * np.triu(np.ones([N_CELLS, N_CELLS]))
    np.fill_diagonal(Q1, -1)
    Q2 = np.zeros([N_CELLS, N_CELLS])
    np.fill_diagonal(Q2, 2)

    line = np.concatenate([Q1] + [Q2] * (N_ATOMS - 1), axis=1)
    for i in range(N_ATOMS):
        Q[i * N_CELLS:i * N_CELLS + N_CELLS, :] = np.roll(line, i * N_CELLS)

    return dimod.BinaryQuadraticModel.from_numpy_matrix(Q)


def objective_Q():
    N = N_ATOMS * N_CELLS
    [x, y] = np.indices((N, N))
    [i, j] = x_index(x)
    [k, l] = x_index(y)
    j_loc = index_to_location(j)
    l_loc = index_to_location(l)
    dists = distance(j_loc, l_loc)
    potentials_full = np.nan_to_num(total_potential(1, dists), 0)
    potentials = 2 * np.triu(potentials_full)

    return potentials
