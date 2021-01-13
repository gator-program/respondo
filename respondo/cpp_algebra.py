import numpy as np

from adcc.functions import zeros_like, empty_like, ones_like
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
    def __init__(self, real=None, imag=None):
        if real is None and imag is None:
            raise ValueError("Either real or imaginary part must be given.")
        self.real = real
        self.imag = imag
        if self.imag is None:
            self.imag = zeros_like(self.real)
        if self.real is None:
            self.real = zeros_like(self.imag)

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
        diagonal_real = self.M.diagonal() 
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
        self.diagonal = self.adcmatrix.diagonal()
        self.projection = projection

    def __matmul__(self, invec):
        eps = 0
        shifted_diagonal = self.diagonal - (self.shift - eps) * ones_like(self.diagonal)

        gamma_diagonal = self.gamma * ones_like(self.diagonal)
        test = shifted_diagonal * shifted_diagonal

        inverse_diagonal = -1.0 * test - (self.gamma * self.gamma) * ones_like(
            self.diagonal
        )
        real_prec = (
            -1.0 * shifted_diagonal * invec.real - 1.0 * gamma_diagonal * invec.imag
        )
        imag_prec = -1.0 * gamma_diagonal * invec.real + shifted_diagonal * invec.imag

        outvec = empty_like(invec)

        outvec.real = real_prec / inverse_diagonal
        outvec.imag = imag_prec / inverse_diagonal
        if self.projection:
            outvec -= self.projection(outvec)
        return outvec
