import scipy.linalg as la
import numpy as np

from adcc.functions import dot, zeros_like, empty_like, ones_like
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from adcc import AmplitudeVector
from adcc import copy
from adcc.solver.conjugate_gradient import State
from adcc.functions import evaluate


class ResponseVectorSymmetrisation:
    def __init__(self, adcmatrix):
        self.isymm = IndexSymmetrisation(adcmatrix)

    def symmetrise(self, x0):
        real = x0.real
        imag = x0.imag
        real = self.isymm.symmetrise(real)
        imag = self.isymm.symmetrise(imag)
        return ResponseVector(real, imag)


class ResponseVector:
    def __init__(self, real, imag=None):
        self.real = real
        if imag is None:
            self.imag = zeros_like(self.real)
        else:
            self.imag = imag

    def dot(self, invec):
        return self.real @ invec.real + self.imag @ invec.imag

    def __matmul__(self, invec):
        return self.dot(invec)

    def __add__(self, invec):
        return ResponseVector(self.real + invec.real, self.imag + invec.imag)

    def __sub__(self, invec):
        return ResponseVector(self.real - invec.real, self.imag - invec.imag)

    def __mul__(self, scalar):
        return ResponseVector(scalar * self.real, scalar * self.imag)

    def __rmul__(self, scalar):
        return ResponseVector(scalar * self.real, scalar * self.imag)

    def __truediv__(self, other):
        return ResponseVector(self.real / other.real, self.imag / other.imag)

    def zeros_like(self):
        return ResponseVector(zeros_like(self.real), zeros_like(self.imag))

    def empty_like(self):
        return ResponseVector(empty_like(self.real), empty_like(self.imag))

    def copy(self):
        return ResponseVector(copy(self.real), copy(self.imag))

    def evaluate(self):
        self.real.evaluate()
        self.imag.evaluate()
        return self


class ComplexPolarizationPropagatorMatrix:
    """
    Hermitian CPP matrix of the form
    (M-w    gamma  )
    (gamma  -(M-w) )
    """
    def __init__(self, adcmatrix, omega=0, gamma=0):
        self.M = adcmatrix
        self.omega = omega
        self.gamma = gamma

    @property
    def shape(self):
        return (2 * self.M.shape[0], 2 * self.M.shape[1])

    @property
    def approximate_diagonal(self):
        diagonal_real = AmplitudeVector(*tuple(
            self.M.diagonal(block) for block in self.M.blocks
        ))
        diagonal_imag = -1.0 * diagonal_real
        return ResponseVector(diagonal_real, diagonal_imag)

    def __matmul__(self, invec):
        real = self.M @ invec.real - self.omega * invec.real + self.gamma * invec.imag
        imag = self.gamma * invec.real - self.M @ invec.imag + self.omega * invec.imag
        ret = ResponseVector(real, imag)
        return ret


class ComplexPolarizationPropagatorPinv:
    """
    Pseudo-inverse for the CPP Matrix
    """
    def __init__(self, matrix, shift, gamma, projection=None):
        self.shift = shift
        self.gamma = gamma
        self.matrix = matrix
        self.adcmatrix = self.matrix.M
        self.diagonal = AmplitudeVector(*tuple(
            self.adcmatrix.diagonal(block) for block in self.adcmatrix.blocks
        ))
        self.projection = projection

    def __matmul__(self, invec):
        eps = 0
        shifted_diagonal = (self.diagonal
                            - (self.shift - eps) * ones_like(self.diagonal))

        gamma_diagonal = self.gamma * ones_like(self.diagonal)
        test = shifted_diagonal*shifted_diagonal

        inverse_diagonal = -1.0 * test - (self.gamma * self.gamma) * ones_like(self.diagonal)
        real_prec = -1.0 * shifted_diagonal * invec.real - 1.0 * gamma_diagonal * invec.imag
        imag_prec = -1.0 * gamma_diagonal * invec.real + shifted_diagonal * invec.imag

        outvec = empty_like(invec)

        outvec.real = real_prec / inverse_diagonal
        outvec.imag = imag_prec / inverse_diagonal
        if self.projection:
            outvec -= self.projection(outvec)
        return outvec


# TODO: currently not used, but could be useful for folded ADC(2) CPP
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
    else:
        x0 = copy(x0)

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
        state.solution = explicit_symmetrisation.symmetrise(state.solution)
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
        state.solution -= evaluate((w - rhs) / D)
        if explicit_symmetrisation:
            state.solution = explicit_symmetrisation.symmetrise(state.solution)
        if projection:
            state.solution -= projection(state.solution)
        state.residual = evaluate(state.solution - old_vec)

        state.residual_norm = np.sqrt(state.residual @ state.residual)
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
