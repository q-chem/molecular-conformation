# Equations from paper
A scratch pad to expand/simplify equations from the paper.
## Equation (8)
Problem hamiltonian
$$
\min H = \sum^B_i \sum^N_j \sum^B_k \sum^N_l U_{ijkl}x_{ij}x_{kl} +  \\
A\sum^B_i(\sum^N_jx_{ij} -1)^2 + 
\\ A\sum^N_j(\sum^B_ix_{ij}(\sum^B_kx_{kj}- 1))
$$

Simplifying line 2 by expanding the square
$$
A\sum^B_i(\sum^N_jx_{ij}^2 + 2\sum^N_j\sum^N_kx_{ij}x_{ik} - 2\sum^N_jx_{ij} + 1)
$$
As $x_{ij}^2 =x_{ij}$
$$
A\sum^B_i(2\sum^N_j\sum^N_kx_{ij}x_{ik} - \sum^N_jx_{ij} + 1)
$$
