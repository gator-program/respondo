from adcc.AmplitudeVector import AmplitudeVector
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from adcc.functions import empty_like, evaluate

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


class ComplexPolarizationPropagatorMatrixFolded:
    def __init__(self, matrix, omega, gamma):
        self.matrix = matrix
        self.omega = omega
        self.gamma = gamma
        self.isymm = IndexSymmetrisation(matrix)

        # intermediates
        self.D22_shifted = evaluate(self.matrix.diagonal().pphh - self.omega)
        self.D22_shifted_squared = evaluate(
            self.D22_shifted * self.D22_shifted + self.gamma**2
        )

    @property
    def shape(self):
        return (2 * self.matrix.shape[0], 2 * self.matrix.shape[1])

    def _apply_D_G(self, other):
        app_ds = self.matrix.block_apply("pphh_ph", other.ph)
        tmp_D_double = app_ds * self.D22_shifted / self.D22_shifted_squared
        self.isymm.symmetrise([AmplitudeVector(pphh=tmp_D_double)])

        tmp_D_single = empty_like(other)
        tmp_D_single.ph = self.matrix.block_apply("ph_pphh", tmp_D_double)

        out_D = (tmp_D_single + other * self.omega).evaluate()

        tmp_G_double = app_ds * self.gamma / self.D22_shifted_squared
        self.isymm.symmetrise([AmplitudeVector(pphh=tmp_G_double)])

        tmp_G_single = empty_like(other)
        tmp_G_single.ph = self.matrix.block_apply("ph_pphh", tmp_G_double)
        out_G = (other * self.gamma + tmp_G_single).evaluate()
        return out_D, out_G

    def unfold_solution(self, other, rhs_full):
        rhs_real = rhs_full.real
        rhs_imag = rhs_full.imag

        tmp_real = self.matrix.block_apply("pphh_ph", other.real.ph)
        tmp_imag = self.matrix.block_apply("pphh_ph", other.imag.ph)

        out_real = (
            self.D22_shifted * (rhs_real.pphh - tmp_real)
            - self.gamma * (rhs_imag.pphh - tmp_imag)
                ) / self.D22_shifted_squared

        self.isymm.symmetrise([AmplitudeVector(pphh=out_real)])

        out_imag = (
            self.D22_shifted * (rhs_imag.pphh - tmp_imag)
            + self.gamma * (rhs_real.pphh - tmp_real)
                ) / self.D22_shifted_squared
        self.isymm.symmetrise([AmplitudeVector(pphh=out_imag)])

        return ResponseVector(AmplitudeVector(ph=other.real.ph, pphh=out_real),
                              AmplitudeVector(ph=other.imag.ph, pphh=out_imag))

    def __matmul__(self, invec):
        tmp_real = empty_like(invec.real)
        tmp_imag = empty_like(invec.imag)
        tmp_real.ph = self.matrix.block_apply("ph_ph", invec.real.ph)
        tmp_imag.ph = self.matrix.block_apply("ph_ph", invec.imag.ph)
        tmp_D_real, tmp_G_real = self._apply_D_G(invec.real)
        tmp_D_imag, tmp_G_imag = self._apply_D_G(invec.imag)

        real = tmp_real - tmp_D_real + tmp_G_imag
        imag = -1.0 * tmp_imag + tmp_D_imag + tmp_G_real

        ret = ResponseVector(real, imag)
        return ret

    def fold_rhs(self, rhs):
        rhs_real = rhs.real
        rhs_imag = rhs.imag

        tmp_real = self.D22_shifted * rhs_real.pphh / self.D22_shifted_squared
        tmp_imag = self.gamma * rhs_imag.pphh / self.D22_shifted_squared
        tmp_double = (tmp_imag - tmp_real).evaluate()
        self.isymm.symmetrise([AmplitudeVector(pphh=tmp_double)])
        tmp_single = self.matrix.block_apply("ph_pphh", tmp_double)
        # needs to be evaluated
        out_real = (rhs_real.ph + tmp_single).evaluate()

        tmp_imag = self.D22_shifted * rhs_imag.pphh / self.D22_shifted_squared
        tmp_real = self.gamma * rhs_real.pphh / self.D22_shifted_squared
        tmp_double = tmp_imag - tmp_real
        self.isymm.symmetrise([AmplitudeVector(pphh=tmp_double)])
        tmp_single = self.matrix.block_apply("ph_pphh", tmp_double)
        # needs to be evaluated
        out_imag = (rhs_imag.ph - tmp_single).evaluate()

        return ResponseVector(AmplitudeVector(ph=out_real),
                              AmplitudeVector(ph=out_imag))


class MatrixFolded:
    def __init__(self, matrix, omega):
        self.matrix = matrix
        self.omega = omega
        self.isymm = IndexSymmetrisation(self.matrix)

    def _apply_D(self, other):
        tmp = self.matrix.block_apply("pphh_ph", other.ph)
        shifted_diagonal = self.matrix.diagonal().pphh - self.omega
        squared_diagonal = (
            shifted_diagonal * shifted_diagonal
        )

        tmp_D_double = tmp * shifted_diagonal / squared_diagonal
        self.isymm.symmetrise([AmplitudeVector(pphh=tmp_D_double)])

        tmp_D_single = empty_like(other)
        tmp_D_single.ph = self.matrix.block_apply("ph_pphh", tmp_D_double)

        out_D = (tmp_D_single + other * self.omega).evaluate()

        return out_D

    def __matmul__(self, invec):
        tmp_real = empty_like(invec)
        tmp_real.ph = self.matrix.block_apply("ph_ph", invec.ph)
        tmp_D_real = self._apply_D(invec)

        real = tmp_real - tmp_D_real
        return real

    def fold_rhs(self, rhs):
        rhs_real = rhs
        shifted_diagonal = self.matrix.diagonal().pphh - self.omega
        squared_diagonal = (
            shifted_diagonal * shifted_diagonal
        )

        tmp_real = shifted_diagonal * rhs_real.pphh / squared_diagonal
        tmp_double = (-1.0 * tmp_real).evaluate()
        self.isymm.symmetrise([AmplitudeVector(pphh=tmp_double)])
        tmp_single = self.matrix.block_apply("ph_pphh", tmp_double)
        # needs to be evaluated
        out_real = (rhs_real.ph + tmp_single).evaluate()

        return AmplitudeVector(ph=out_real)

    def unfold_solution(self, other, rhs_full):
        rhs_real = rhs_full
        shifted_diagonal = self.matrix.diagonal().pphh - self.omega
        squared_diagonal = (
            shifted_diagonal * shifted_diagonal
        )
        tmp_real = self.matrix.block_apply("pphh_ph", other.ph)

        out_real = (
            shifted_diagonal * (rhs_real.pphh - tmp_real)
                ) / squared_diagonal

        self.isymm.symmetrise([AmplitudeVector(pphh=out_real)])

        return AmplitudeVector(ph=other.ph, pphh=out_real)


class MatrixWrapper:
    def __init__(self, matrix, omega, gamma, fold_doubles):
        self.matrix = matrix
        self.omega = omega
        self.gamma = gamma
        self.fold_doubles = fold_doubles
        self._fold_rhs = None
        self._unfold_solution = None
        if self.fold_doubles:
            assert self.matrix.method.level == 2
        self.__select_matrix()

    def __str__(self):
        info = f"MatrixWrapper(omega={self.omega}, gamma={self.gamma}, "
        info += f"fold_doubles={self.fold_doubles})"
        return info

    def __select_matrix(self):
        if self.gamma == 0.0:
            # TODO: implement shift correctly
            if self.fold_doubles:
                self._wrapped = MatrixFolded(self.matrix, self.omega)
                self._precond = JacobiPreconditioner(self.matrix, self.omega)
                self._precond.diagonal = self.matrix.diagonal().ph
                self._fold_rhs = self._wrapped.fold_rhs
                self._unfold_solution = self._wrapped.unfold_solution
                self._symm = None
            else:
                self._wrapped = self.matrix
                self._precond = JacobiPreconditioner(self.matrix, self.omega)
                self._symm = IndexSymmetrisation(self.matrix)
        else:
            if self.fold_doubles:
                self._wrapped = ComplexPolarizationPropagatorMatrixFolded(
                    self.matrix, self.omega, self.gamma
                )
                self._precond = ComplexPolarizationPropagatorPinv(
                    AmplitudeVector(ph=self.matrix.diagonal().ph),
                    self.omega, self.gamma
                )
                self._fold_rhs = self._wrapped.fold_rhs
                self._unfold_solution = self._wrapped.unfold_solution
                self._symm = None
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

    def form_solution(self, solution, rhs):
        if callable(self._unfold_solution):
            return self._unfold_solution(solution, rhs)
        else:
            return solution

    def form_rhs(self, rhs):
        if callable(self._fold_rhs):
            return self._fold_rhs(rhs)
        else:
            return rhs
