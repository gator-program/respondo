import numpy as np
from adcc import AmplitudeVector
from adcc.adc_pp.modified_transition_moments import modified_transition_moments
from adcc.Excitation import Excitation
from adcc.functions import einsum
from adcc.workflow import construct_adcmatrix

from .misc import select_property_method
from .solve_response import solve_response, transition_polarizability


def mcd_bterm(state, property_method=None, **solver_args):
    if not isinstance(state, Excitation):
        raise TypeError()
    matrix = construct_adcmatrix(state.parent_state.matrix)
    if property_method is None:
        property_method = select_property_method(matrix)
    hf = matrix.reference_state
    mp = matrix.ground_state

    dips_el = hf.operators.electric_dipole
    dips_mag = hf.operators.magnetic_dipole
    rhss_el = modified_transition_moments(property_method, mp, dips_el)
    rhss_mag = modified_transition_moments(property_method, mp, dips_mag)

    # the minus sign is required due to the anti-hermiticity of the magnetic dipole operator
    response_mag = [
        -1.0 * solve_response(matrix, rhs_mag, omega=0.0, gamma=0.0, **solver_args) for rhs_mag in rhss_mag
    ]

    v_f = state.excitation_vector
    e_f = state.excitation_energy

    # TODO: folding?
    # project the f-th state out of the matrix-vector product
    def projection(X, bl=None):
        if bl:
            vb = getattr(v_f, bl)
            return vb * (vb.dot(X)) / (vb.dot(vb))
        else:
            return v_f * (v_f @ X) / (v_f @ v_f)

    # matrix_shifted.projection = projection
    response_el = [
        solve_response(matrix, rhs_el, omega=e_f, gamma=0.0, projection=projection, **solver_args)
        for rhs_el in rhss_el
    ]

    term1 = transition_polarizability(property_method, mp, v_f, dips_el, response_mag)
    term2 = transition_polarizability(property_method, mp, v_f, dips_mag, response_el)

    # Levi-Civita tensor
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1

    tdip_f = state.transition_dipole_moment
    # the minus sign accounts for the negative charge, since it is not included in the operators
    # TODO as soon as PR #190 in adcc is merged: remove minus
    B = -1.0 * np.einsum("abc,a,bc->", epsilon, tdip_f, term1 + np.transpose(term2))
    return B
