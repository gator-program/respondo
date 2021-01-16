from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from .cpp_algebra import ResponseVector, ResponseVectorSymmetrisation


class ComplexPolarizationPropagatorMatrix:
    def __init__(self, matrix, omega, gamma):
        self.matrix = matrix
        self.omega = omega
        self.gamma = gamma

    @property
    def shape(self):
        return (2 * self.matrix.shape[0], 2 * self.matrix.shape[1])

    def __matmul__(self, invec):
        real = (
            self.matrix @ invec.real - self.omega * invec.real
            + self.gamma * invec.imag
        )
        imag = (
            self.gamma * invec.real - self.matrix @ invec.imag
            + self.omega * invec.imag
        )
        ret = ResponseVector(real, imag)
        return ret


class ComplexPolarizationPropagatorPinv:
    """
    Pseudo-inverse for the CPP Matrix
    """

    def __init__(self, diagonal, omega, gamma, projection=None):
        self.omega = omega
        self.gamma = gamma
        self.diagonal = diagonal
        self.projection = projection

    def __matmul__(self, invec):
        shifted_diagonal = self.diagonal - self.omega

        d2 = shifted_diagonal * shifted_diagonal
        g2 = self.gamma * self.gamma

        denominator = -1.0 * d2 - g2
        real_prec = (
            -1.0 * shifted_diagonal * invec.real - self.gamma * invec.imag
        )
        imag_prec = (
            -1.0 * self.gamma * invec.real + shifted_diagonal * invec.imag
        )

        outvec = ResponseVector(
            real=real_prec / denominator,
            imag=imag_prec / denominator
        )
        if self.projection:
            outvec -= self.projection(outvec)
        return outvec


class MatrixWrapper:
    def __init__(self, matrix, omega, gamma, fold_doubles):
        self.matrix = matrix
        self.omega = omega
        self.gamma = gamma
        self.fold_doubles = fold_doubles
        if self.fold_doubles:
            assert self.matrix.method.level == 2
        self.__select_matrix()

    def __select_matrix(self):
        if self.gamma == 0.0:
            if self.fold_doubles:
                raise NotImplementedError("")
            else:
                self._wrapped = self.matrix
                self._precond = JacobiPreconditioner(self.matrix, self.omega)
                self._symm = IndexSymmetrisation(self.matrix)
        else:
            if self.fold_doubles:
                raise NotImplementedError("")
            else:
                self._wrapped = ComplexPolarizationPropagatorMatrix(
                    self.matrix, self.omega, self.gamma
                )
                self._precond = ComplexPolarizationPropagatorPinv(
                    self.matrix.diagonal(), self.omega, self.gamma
                )
                self._symm = ResponseVectorSymmetrisation(self.matrix)

    @property
    def preconditioner(self):
        return self._precond

    @property
    def explicit_symmetrisation(self):
        return self._symm

    def __matmul__(self, invec):
        return self._wrapped @ invec

    @property
    def shape(self):
        return self._wrapped.shape

    def form_solution(self, solution):
        raise NotImplementedError()

    def form_rhs(self, rhs):
        raise NotImplementedError()
