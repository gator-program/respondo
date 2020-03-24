import sys
import scipy.linalg as la
import numpy as np

from adcc.functions import dot, zeros_like, empty_like, ones_like, multiply, divide
from adcc.solver.preconditioner import PreconditionerIdentity
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from adcc import AmplitudeVector
from adcc import copy


class ResponseVectorSymmetrisation:
    def __init__(self, adcmatrix):
        self.isymm = IndexSymmetrisation(adcmatrix)

    def symmetrise(self, x0, tmp):
        real = x0.real
        imag = x0.imag
        self.isymm.symmetrise([real], [tmp.real])
        self.isymm.symmetrise([imag], [tmp.imag])
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


class CppSolverState:
    def __init__(self):
        self.solution = None             # Current approximation to the solution
        self.residual = None             # Current residual
        self.residual_norm = None        # Current residual norm
        self.all_residual_norms = []
        self.converged = False           # Flag whether iteration is converged
        self.n_iter = 0                  # Number of iterations
        self.n_applies = 0               # Number of applies


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


def conjugate_gradient(matrix, rhs, x0=None, conv_tol=1e-9, max_iter=100,
                       callback=None, Pinv=None, cg_type="polak_ribiere",
                       explicit_symmetrisation=IndexSymmetrisation):
    """An implementation of the conjugate gradient algorithm.

    This algorithm implements the "flexible" conjugate gradient using the
    Polak-Ribière formula, but allows to employ the "traditional"
    Fletcher-Reeves formula as well.
    It solves `matrix @ x = rhs` for `x` by minimising the residual
    `matrix @ x - rhs`.

    Parameters
    ----------
    matrix
        Matrix object. Should be an ADC matrix.
    rhs
        Right-hand side, source.
    x0
        Initial guess
    conv_tol : float
        Convergence tolerance on the l2 norm of residuals to consider
        them converged.
    max_iter : int
        Maximum number of iterations
    callback
        Callback to call after each iteration
    Pinv
        Preconditioner to A, typically an estimate for A^{-1}
    cg_type : string
        Identifier to select between polak_ribiere and fletcher_reeves
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

    # The problem size
    n_problem = matrix.shape[1]

    if x0 is None:
        # Start with random guess
        raise NotImplementedError("Random guess is not yet implemented.")
        x0 = np.random.rand((n_problem))
    else:
        x0 = copy(x0)

    if Pinv is None:
        Pinv = PreconditionerIdentity()
    if Pinv is not None and isinstance(Pinv, type):
        Pinv = Pinv(matrix)

    def is_converged(state):
        state.converged = state.residual_norm < conv_tol
        return state.converged

    state = CppSolverState()

    # Initialise iterates
    state.solution = x0
    state.residual = rhs - matrix @ state.solution
    state.n_applies += 1
    state.residual_norm = np.sqrt(state.residual @ state.residual)
    pk = zk = Pinv @ state.residual

    if explicit_symmetrisation:
        # TODO Not sure this is the right spot ... also this syntax is ugly
        pk = explicit_symmetrisation.symmetrise(pk, x0)

    callback(state, "start")
    while state.n_iter < max_iter:
        state.n_iter += 1

        # Update ak and iterated solution
        # TODO This needs to be modified for general optimisations,
        #      i.e. where A is non-linear
        # https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
        Apk = matrix @ pk
        state.n_applies += 1
        res_dot_zk = dot(state.residual, zk)
        ak = float(res_dot_zk / dot(pk, Apk))
        state.solution += ak * pk

        residual_old = state.residual
        state.residual = residual_old - ak * Apk
        state.residual_norm = np.sqrt(state.residual @ state.residual)
        state.all_residual_norms.append(state.residual_norm)

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations (== "
                                 + str(max_iter) + " reached in conjugate "
                                 "gradient procedure.")

        zk = Pinv @ state.residual

        if explicit_symmetrisation:
            # TODO Not sure this is the right spot ... also this syntax is ugly
            zk = explicit_symmetrisation.symmetrise(zk, pk)

        if cg_type == "fletcher_reeves":
            bk = float(dot(zk, state.residual) / res_dot_zk)
        elif cg_type == "polak_ribiere":
            bk = float(dot(zk, (state.residual - residual_old)) / res_dot_zk)
        pk = zk + bk * pk


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
        test = zeros_like(shifted_diagonal)
        multiply(shifted_diagonal, shifted_diagonal, test)

        inverse_diagonal = -1.0 * test - (self.gamma * self.gamma) * ones_like(self.diagonal)
        real_prec = -1.0 * shifted_diagonal * invec.real - 1.0 * gamma_diagonal * invec.imag
        imag_prec = -1.0 * gamma_diagonal * invec.real + shifted_diagonal * invec.imag

        outvec = empty_like(invec)

        divide(real_prec, inverse_diagonal, outvec.real)
        divide(imag_prec, inverse_diagonal, outvec.imag)
        if self.projection:
            outvec -= self.projection(outvec)
        return outvec
