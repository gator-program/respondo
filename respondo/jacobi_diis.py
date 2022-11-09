#!/usr/bin/env python3
import sys
from itertools import product
import numpy as np
import scipy.linalg as la

from adcc import copy, evaluate
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation


class State:
    def __init__(self):
        self.solution = None       # Current approximation to the solution
        self.residual = None       # Current residual
        self.residual_norm = None  # Current residual norm
        self.converged = False     # Flag whether iteration is converged
        self.n_iter = 0            # Number of iterations
        self.n_applies = 0         # Number of applies


def default_print(state, identifier, file=sys.stdout):
    if identifier == "start" and state.n_iter == 0:
        print("Niter residual_norm", file=file)
    elif identifier == "next_iter":
        fmt = "{n_iter:3d}  {residual:12.5g}"
        print(fmt.format(n_iter=state.n_iter,
                         residual=np.max(state.residual_norm)), file=file)
    elif identifier == "is_converged":
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies)


def jacobi_diis(matrix, rhs, x0, Dinv, conv_tol=1e-9, max_iter=100,
                callback=None, explicit_symmetrisation=IndexSymmetrisation,
                max_subspace=8):
    """An implementation of the Jacobi-DIIS algorithm.
    It solves `matrix @ x = rhs` for `x` by minimising the residual
    `matrix @ x - rhs` using a Jacobi update with DIIS acceleration.
    Parameters
    ----------
    matrix
        Matrix object. Should be an ADC matrix.
    rhs
        Right-hand side, source.
    x0
        Initial guess
    Dinv
        Inverse of the matrix diagonal D^{-1}
    conv_tol : float
        Convergence tolerance on the l2 norm of residuals to consider
        them converged.
    max_iter : int
        Maximum number of iterations
    callback
        Callback to call after each iteration
    explicit_symmetrisation
        Explicit symmetrisation to perform during iteration to ensure
        obtaining an eigenvector with matching symmetry criteria.
    """
    if callback is None:
        def callback(state, identifier):
            pass

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if x0 is None:
        raise NotImplementedError("Random guess is not yet implemented.")
    else:
        x0 = copy(x0)

    if Dinv is not None and isinstance(Dinv, type):
        Dinv = Dinv(matrix)

    def is_converged(state):
        state.converged = state.residual_norm < conv_tol
        return state.converged

    state = State()
    # Initialise iterates
    state.solution = x0
    if explicit_symmetrisation:
        state.solution = explicit_symmetrisation.symmetrise(state.solution)
    state.residual = evaluate(rhs - matrix @ state.solution)
    state.residual_norm = np.sqrt(state.residual @ state.residual)

    callback(state, "start")
    diis = DIIS(max_subspace=max_subspace)
    while state.n_iter < max_iter:
        state.n_iter += 1

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations (== "
                                 + str(max_iter) + " reached in Jacobi/DIIS "
                                 "procedure.")

        x_new = evaluate(state.solution + Dinv @ state.residual)
        if explicit_symmetrisation:
            x_new = explicit_symmetrisation.symmetrise(x_new)
        state.solution = diis.extrapolate(x_new, x_new - state.solution)

        state.residual = evaluate(rhs - matrix @ state.solution)
        state.n_applies += 1
        state.residual_norm = np.sqrt(state.residual @ state.residual)


class DIIS:
    # adapted from 
    # https://github.com/edeprince3/pdaggerq/blob/master/examples/full_cc_codes/diis.py
    def __init__(self, max_subspace, start_iter=3):
        self.nvecs = max_subspace
        self.error_vecs = []
        self.prev_vecs = []
        self.start_iter = start_iter
        self.iter = 0

    def extrapolate(self, iterate, error):
        if self.iter < self.start_iter:
            self.iter += 1
            return iterate

        self.prev_vecs.append(iterate)
        self.error_vecs.append(error)
        self.iter += 1

        if len(self.prev_vecs) > self.nvecs:
            self.prev_vecs.pop(0)
            self.error_vecs.pop(0)

        b_mat, rhs = self.get_bmatrix()
        c = np.linalg.solve(b_mat, rhs)
        c = c.flatten()

        new_iterate = self.prev_vecs[0].zeros_like()
        for ii in range(len(self.prev_vecs)):
            new_iterate += c[ii] * self.prev_vecs[ii]
        return new_iterate.evaluate()

    def get_bmatrix(self):
        dim = len(self.prev_vecs)
        b = np.zeros((dim, dim))
        for i, j in product(range(dim), repeat=2):
            if i <= j:
                b[i, j] = self.error_vecs[i].dot(self.error_vecs[j])
                b[j, i] = b[i, j]
        b = np.hstack((b, -1 * np.ones((dim, 1))))
        b = np.vstack((b, -1 * np.ones((1, dim + 1))))
        b[-1, -1] = 0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1, 0] = -1
        return b, rhs