import adcc
import numpy as np
import scipy.linalg as la

from adcc.AmplitudeVector import AmplitudeVector
from adcc.solver import IndexSymmetrisation

# TODO: remove...


class State:
    def __init__(self):
        self.solution = None             # Current approximation to the solution
        self.residual = None             # Current residual
        self.residual_norm = None        # Current residual norm
        self.all_residual_norms = []
        self.converged = False           # Flag whether iteration is converged
        self.n_iter = 0                  # Number of iterations
        self.n_applies = 0               # Number of applies


def jacobi(matrix, rhs, x0=None, conv_tol=1e-9, max_iter=100,
           callback=None, explicit_symmetrisation=IndexSymmetrisation,
           diis=False, max_error_vectors=10, projection=None):
    """An implementation of the Jacobi-DIIS solver"""
    if callback is None:
        def callback(state, identifier):
            pass

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    # The problem size
    n_problem = matrix.shape[1]

    if x0 is None:
        # Start with random guess
        raise NotImplementedError("Random guess is not yet implemented.")
        x0 = np.random.rand((n_problem))
    else:
        x0 = adcc.copy(x0)

    def is_converged(state):
        state.converged = state.residual_norm < conv_tol
        return state.converged

    state = State()

    D = AmplitudeVector(*tuple(
        matrix.diagonal(block) for block in matrix.blocks
    ))

    state.solution = x0
    old_vec = state.solution.copy()
    w = matrix @ state.solution
    state.solution -= (w - rhs) / D
    if explicit_symmetrisation:
        state.solution = explicit_symmetrisation.symmetrise([state.solution], [rhs])[0]
    if projection:
        state.solution -= projection(state.solution)
    state.residual = state.solution - old_vec

    residuals = []
    solutions = []

    callback(state, "start")
    while state.n_iter < max_iter:
        state.n_iter += 1
        state.n_applies += 1

        old_vec = state.solution.copy()
        w = matrix @ state.solution
        state.solution -= (w - rhs) / D
        if explicit_symmetrisation:
            state.solution = explicit_symmetrisation.symmetrise([state.solution], [rhs])[0]
        if projection:
            state.solution -= projection(state.solution)
        state.residual = state.solution - old_vec

        state.residual_norm = np.sqrt(state.residual @ state.residual)
        state.all_residual_norms.append(state.residual_norm)
        # DIIS
        if diis:
            if len(residuals) >= max_error_vectors:
                residuals.pop(0)
                solutions.pop(0)
            residuals.append(state.residual)
            solutions.append(state.solution)
            if len(residuals) >= 3:
                diis_size = len(residuals) + 1
                A = np.zeros((diis_size, diis_size))
                A[:, 0] = -1.0
                A[0, :] = -1.0
                # TODO: only compute upper/lower triangle
                for i, r1 in enumerate(residuals, 1):
                    for j, r2 in enumerate(residuals, 1):
                        A[i, j] = r1 @ r2
                diis_rhs = np.zeros(diis_size)
                diis_rhs[0] = -1.0
                weights = np.linalg.solve(A, diis_rhs)[1:]
                state.solution = adcc.zeros_like(state.solution)
                for i, s in enumerate(solutions):
                    state.solution += weights[i] * s

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations (== "
                                 + str(max_iter) + " reached in Jacobi "
                                 "procedure.")
