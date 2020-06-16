from adcc import AdcMatrix, LazyMp

import numpy as np

from adcc.modified_transition_moments import (
    compute_modified_transition_moments,
)

from adcc.solver.conjugate_gradient import default_print
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver import IndexSymmetrisation
from adcc.solver.conjugate_gradient import conjugate_gradient

from .cpp_algebra import (
    ResponseVector,
    ComplexPolarizationPropagatorPinv,
    ResponseVectorSymmetrisation,
)
from .cpp_algebra import ComplexPolarizationPropagatorMatrix as CppMatrix

_comps = ["x", "y", "z"]


def compute_static_polarizability(
    matrix_method, reference_state, **solver_args
):
    """
    Compute the static polarizability of the electronic
    ground state.
    """
    dips = reference_state.operators.electric_dipole
    ground_state = LazyMp(reference_state)
    matrix = AdcMatrix(matrix_method, ground_state)
    rhss = [
        compute_modified_transition_moments(matrix, dip, "adc2")
        for dip in dips
    ]

    preconditioner = JacobiPreconditioner(matrix)
    explicit_symmetrisation = IndexSymmetrisation(matrix)
    preconditioner.update_shifts(0.0)

    solutions = []
    for mu in range(3):
        rhs = rhss[mu]
        x0 = preconditioner.apply(rhs)
        print(f"Solving response equation for component {_comps[mu]}.")
        res = conjugate_gradient(
            matrix,
            rhs=rhs,
            x0=x0,
            callback=default_print,
            explicit_symmetrisation=explicit_symmetrisation,
            **solver_args,
        )
        solutions.append(res.solution)

    polarizability = np.zeros((3, 3))
    for c1 in range(3):
        for c2 in range(c1, 3):
            polarizability[c1, c2] = 2.0 * solutions[c2] @ rhss[c1]
            polarizability[c2, c1] = polarizability[c1, c2]
    return polarizability


def compute_complex_polarizability(
    matrix_method,
    reference_state,
    omega=0.0,
    gamma=0.0,
    solver=conjugate_gradient,
    **solver_args,
):
    """
    Compute the complex frequency-dependent polarizability of the electronic
    ground state.
    """
    # TODO: allow for multiple frequencies from outside, multi-frequency solver
    dips = reference_state.operators.electric_dipole
    ground_state = LazyMp(reference_state)
    matrix = AdcMatrix(matrix_method, ground_state)
    rhss = [
        compute_modified_transition_moments(matrix, dip, "adc2")
        for dip in dips
    ]
    cpp_matrix = CppMatrix(matrix, gamma=gamma, omega=omega)
    Pinv = ComplexPolarizationPropagatorPinv(
        cpp_matrix, shift=omega, gamma=gamma
    )
    solutions = []
    for mu in range(3):
        rhs = ResponseVector(rhss[mu])
        x0 = Pinv @ rhs
        rsymm = ResponseVectorSymmetrisation(matrix)
        guess_symm = rsymm.symmetrise(x0)

        cpp_matrix.omega = omega
        Pinv.shift = omega
        print(f"Solving complex response equation for component {_comps[mu]}.")
        res1 = solver(
            cpp_matrix,
            rhs=rhs,
            x0=guess_symm,
            callback=default_print,
            Pinv=Pinv,
            explicit_symmetrisation=rsymm,
            **solver_args,
        )
        cpp_matrix.omega = -omega
        Pinv.shift = -omega
        res2 = solver(
            cpp_matrix,
            rhs=rhs,
            x0=guess_symm,
            callback=default_print,
            Pinv=Pinv,
            explicit_symmetrisation=rsymm,
            **solver_args,
        )
        cpp_matrix.omega = omega
        Pinv.shift = omega
        solutions.append((res1.solution, res2.solution))

    polarizability = np.zeros((3, 3), dtype=np.complex)
    for c1 in range(3):
        for c2 in range(c1, 3):
            sol1 = solutions[c2][0]
            sol2 = solutions[c2][1]
            polarizability.real[c1, c2] = (
                sol1.real @ rhss[c1] + sol2.real @ rhss[c1]
            )
            polarizability.imag[c1, c2] = (
                sol1.imag @ rhss[c1] - sol2.imag @ rhss[c1]
            )
            polarizability[c2, c1] = polarizability[c1, c2]
    return polarizability


def one_photon_absorption_cross_section(polarizability, omegas):
    isotropic_avg_im_alpha = (
        1.0 / 3.0 * np.trace(polarizability.imag, axis1=1, axis2=2)
    )
    return 4.0 * np.pi / 137.0 * omegas * isotropic_avg_im_alpha


def compute_c6_dispersion_coefficient(
    matrix_method, reference_state, **solver_args
):
    """
    Compute the ground state C6 dispersion coefficient by quadrature
    """
    points, weights = np.polynomial.legendre.leggauss(12)
    w0 = 0.3
    freqs = w0 * (1 - points) / (1 + points)
    alphas_iso = []

    for w in freqs:
        pol = compute_complex_polarizability(
            matrix_method, reference_state, omega=0.0, gamma=w, **solver_args
        )
        alphas_iso.append(1.0 / 3.0 * np.trace(pol.real))
    alphas_iso = np.array(alphas_iso)
    derivative = w0 * 2 / (1 + points) ** 2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3.0 * integral / np.pi
    return c6
