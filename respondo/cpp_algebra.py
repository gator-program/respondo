from adcc.functions import zeros_like, empty_like
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from adcc import copy
import numpy as np


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
        real = self.real @ invec.real - self.imag @ invec.imag
        imag = self.real @ invec.imag + self.imag @ invec.real
        return np.complex(real, imag)

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
