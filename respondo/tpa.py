import numpy as np

from adcc.adc_pp import modified_transition_moments
from adcc.workflow import construct_adcmatrix
from adcc.Excitation import Excitation
from .cpp_algebra import ResponseVector
from .misc import select_property_method
from .solve_response import solve_response, transition_polarizability


def tpa_resonant(state, property_method=None, **solver_args):
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
            matrix, rhs, omega=state.excitation_energy_uncorrected / 2.0,
            gamma=0.0, **solver_args
        )
        for rhs in rhss
    ]
    # build TPA transition polarizability matrix S for final state f
    pols = transition_polarizability(
        property_method, matrix.ground_state,
        response, dips, state.excitation_vector
    )
    S = pols + pols.T
    # Resonant TPA transition strength
    strength = (
        1.0 / 15.0 * (
            + np.einsum("mm,vv->", S, S)
            + np.einsum("mv,mv->", S, S)
            + np.einsum("mv,vm->", S, S)
        )
    )
    return strength, S