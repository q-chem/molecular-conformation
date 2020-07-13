# Molecular Conformation Problem D-Wave Implementation
Implementing the problem molecular conformation model described in [Quantum computing based hybrid solution strategies for large-scale discrete-continuous optimization problems](https://doi.org/10.1016/j.compchemeng.2019.106630) using D-Wave tools.

## Installation
* Install the required packages using the `Pipfile` or `requirements.txt`
* Fill in your D-Wave token in `dwave.conf.template` and rename it to `dwave.conf`

## Notes
* The hamiltonian is built up using `numpy`
* `qbsolv` is used for the solving which can take a number of different classical and quantum solvers
  * using the CLI the following are supported (using the `-s` option)
    * `tabu` - a classical solver using the TABU algorithm
    * `hybrid` - D-Wave's `LeapHybridSampler()`
    * `embed` - Auto embedded quantum sampler `EmbeddingComposite(DWaveSampler())`
    * `sim_anneal` - classical. `SimulatedAnnealingSampler()` from `dwave-neal`