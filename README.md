# Lanczos_Krylov.jl
Implementation of the Lanczos algorithm in Julia.
(https://en.wikipedia.org/wiki/Lanczos_algorithm)

Evaluates $e^{M} v$ for a matrix $M$ and vector $v$ using a Krylov subspace.

Krylov.jl contains a function simply implementing the algorithm for a given dimension.

Krylov_Adaptive.jl contains a function to determine the optimal dimension for the projection,
then a function implementing the algorithm for the given dimension.
It also contains functions to test the algorithm on time evolution in a quantum system.
