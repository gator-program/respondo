from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
from adcc.adc_pp.modified_transition_moments import modified_transition_moments
from adcc.Excitation import Excitation
from adcc.workflow import construct_adcmatrix

from .cpp_algebra import ResponseVector
from .solve_response import solve_response, transition_polarizability_complex
from .misc import select_property_method

import numpy as np

_comps = ["x", "y", "z"]


def rixs_scattering_strength(F, omega, omega_prime, theta=90 * np.pi / 180):
    strength = 0j
    cc = np.conj
    for A in range(3):
        for B in range(3):
            strength += (2 - 0.5 * np.sin(theta) ** 2) * F[A, B] * cc(F[A, B]) + (
                3.0 / 4.0 * np.sin(theta) ** 2
            ) * (F[A, B] * cc(F[B, A]) + F[A, A] * cc(F[B, B]))
    strength *= omega_prime / omega * 1.0 / 15.0
    return strength


def rixs(state, omega, gamma, property_method=None, rotating_wave=True, **solver_args):
    if not isinstance(state, Excitation):
        raise TypeError()
    matrix = construct_adcmatrix(state.parent_state.matrix)
    if property_method is None:
        property_method = select_property_method(matrix)
    hf = matrix.reference_state
    mp = matrix.ground_state
    dips = hf.operators.electric_dipole
    rhss = modified_transition_moments(property_method, mp, dips)

    response = [
        solve_response(
            matrix, ResponseVector(rhs),
            omega, gamma, **solver_args
        )
        for rhs in rhss
    ]
    # build RIXS transition polarizatbilty F for final state
    F = transition_polarizability_complex(
        property_method,
        mp,
        ResponseVector(state.excitation_vector),
        dips,
        response,
    )
    omega_prime = omega - state.excitation_energy_uncorrected
    # TODO: tests
    if not rotating_wave:
        response_prime = [
            solve_response(
                matrix, ResponseVector(rhs), -omega_prime, -gamma, **solver_args
            )
            for rhs in rhss
        ]
        F_prime = transition_polarizability_complex(
            property_method,
            mp,
            ResponseVector(state.excitation_vector),
            dips,
            response_prime,
        )
        F += F_prime.T
    else:
        mom_product = np.einsum(
            "A,B->AB", state.transition_dipole_moment, mp.dipole_moment(property_method.level)
        )
        gs_term = +mom_product / (-(omega + 1j*gamma)) - mom_product.T / (
            omega_prime + 1j*gamma
        )
        F += gs_term

    strength = rixs_scattering_strength(F, omega, omega_prime)
    return strength, F
