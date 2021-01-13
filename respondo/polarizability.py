from adcc import AdcMatrix, LazyMp

import numpy as np

from adcc.adc_pp import modified_transition_moments
from .cpp_algebra import ResponseVector
from .solve_response import solve_complex, solve_real

_comps = ["x", "y", "z"]


def static_polarizability(matrix_method, reference_state, **solver_args):
    """
    Compute the static polarizability of the electronic
    ground state.
    """
    dips = reference_state.operators.electric_dipole
    ground_state = LazyMp(reference_state)
    matrix = AdcMatrix(matrix_method, ground_state)

    if matrix.method.level < 3:
        property_method = matrix.method
    else:
        # Auto-select second-order properties for third-order calc
        property_method = matrix.method.at_level(2)

    rhss = modified_transition_moments(property_method, ground_state, dips)
    response = [solve_real(matrix, rhs, 0.0, **solver_args) for rhs in rhss]

    polarizability = np.zeros((3, 3))
    for A in range(3):
        for B in range(A, 3):
            polarizability[A, B] = 2.0 * response[B] @ rhss[A]
            polarizability[B, A] = polarizability[A, B]
    return polarizability


def complex_polarizability(
    matrix_method, reference_state, omega=0.0, gamma=0.0, **solver_args
):
    """
    Compute the complex frequency-dependent polarizability of the electronic
    ground state.
    """
    # TODO: allow for multiple frequencies from outside, multi-frequency solver
    dips = reference_state.operators.electric_dipole
    ground_state = LazyMp(reference_state)
    matrix = AdcMatrix(matrix_method, ground_state)

    if matrix.method.level < 3:
        property_method = matrix.method
    else:
        # Auto-select second-order properties for third-order calc
        property_method = matrix.method.at_level(2)

    rhss = modified_transition_moments(property_method, ground_state, dips)
    response_positive = [
        solve_complex(matrix, ResponseVector(rhs), omega, gamma, **solver_args)
        for rhs in rhss
    ]
    response_negative = [
        solve_complex(matrix, ResponseVector(rhs), -omega, gamma, **solver_args)
        for rhs in rhss
    ]

    polarizability = np.zeros((3, 3), dtype=np.complex)
    for A in range(3):
        for B in range(A, 3):
            rsp_pos = response_positive[B]
            rsp_neg = response_negative[B]
            polarizability.real[A, B] = rsp_pos.real @ rhss[A] + rsp_neg.real @ rhss[A]
            polarizability.imag[A, B] = rsp_pos.imag @ rhss[A] - rsp_neg.imag @ rhss[A]
            polarizability[B, A] = polarizability[A, B]
    return polarizability


def one_photon_absorption_cross_section(polarizability, omegas):
    isotropic_avg_im_alpha = 1.0 / 3.0 * np.trace(polarizability.imag, axis1=1, axis2=2)
    return 4.0 * np.pi / 137.0 * omegas * isotropic_avg_im_alpha


def c6_dispersion_coefficient(matrix_method, reference_state, **solver_args):
    """
    Compute the ground state C6 dispersion coefficient by quadrature
    """
    points, weights = np.polynomial.legendre.leggauss(12)
    w0 = 0.3
    freqs = w0 * (1 - points) / (1 + points)
    alphas_iso = []

    for w in freqs:
        pol = complex_polarizability(
            matrix_method, reference_state, omega=0.0, gamma=w, **solver_args
        )
        alphas_iso.append(1.0 / 3.0 * np.trace(pol.real))
    alphas_iso = np.array(alphas_iso)
    derivative = w0 * 2 / (1 + points) ** 2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3.0 * integral / np.pi
    return c6
