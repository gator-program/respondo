import numpy as np
from adcc import AmplitudeVector
from adcc.adc_pp.modified_transition_moments import modified_transition_moments
from adcc.Excitation import Excitation
from adcc.functions import einsum
from adcc.workflow import construct_adcmatrix

from .misc import select_property_method
from .solve_response import solve_response, transition_polarizability


def compute_adc1_f1_mag(magdip, ground_state):
    mtm = magdip.ov + einsum("ijab,jb->ia", ground_state.t2("o1o1v1v1"), magdip.ov)
    return AmplitudeVector(ph=mtm)


def compute_adc2_f1_mag(magdip, ground_state):
    t2 = ground_state.t2("o1o1v1v1")
    td2 = ground_state.td2("o1o1v1v1")
    p0 = ground_state.mp2_diffdm
    d = magdip
    return (
        d.ov
        + 1.0 * einsum("ijab,jb->ia", t2, d.ov + 0.5 * einsum("jkbc,kc->jb", t2, d.ov))
        + 0.5 * (einsum("ij,ja->ia", p0.oo, d.ov) - 1.0 * einsum("ib,ab->ia", d.ov, p0.vv))
        - 1.0 * einsum("ib,ab->ia", p0.ov, d.vv)
        - 1.0 * einsum("ij,ja->ia", d.oo, p0.ov)
        + 1.0 * einsum("ijab,jb->ia", td2, d.ov)
    )


def compute_adc2_f2_mag(magdip, ground_state):
    t2 = ground_state.t2("o1o1v1v1")
    term1 = -1.0 * einsum("ijac,bc->ijab", t2, magdip.vv)
    term2 = -1.0 * einsum("ik,kjab->ijab", magdip.oo, t2)
    term1 = term1.antisymmetrise(2, 3)
    term2 = term2.antisymmetrise(0, 1)
    return term1 - term2


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
    # Electric dipole right-hand-side
    rhss_el = modified_transition_moments(property_method, mp, dips_el)

    # Magnetic dipole right-hand-side
    # TODO: temporary hack, add to adcc...
    if property_method.name == "adc0":
        rhss_mag = modified_transition_moments(property_method, mp, dips_mag)
        rhss_mag = [-1.0 * rhs_mag for rhs_mag in rhss_mag]
    elif property_method.name == "adc1":
        rhss_mag = [-1.0 * compute_adc1_f1_mag(mag, mp) for mag in dips_mag]
    elif property_method.name == "adc2":
        rhss_mag = [
            -1.0
            * AmplitudeVector(
                ph=compute_adc2_f1_mag(mag, mp),
                pphh=compute_adc2_f2_mag(mag, mp),
            )
            for mag in dips_mag
        ]
    else:
        raise NotImplementedError("")

    response_mag = [
        solve_response(matrix, rhs_mag, omega=0.0, gamma=0.0, **solver_args) for rhs_mag in rhss_mag
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

    term1 = -transition_polarizability(property_method, mp, response_mag, dips_el, v_f)
    term2 = -transition_polarizability(property_method, mp, v_f, dips_mag, response_el)

    # Levi-Civita tensor
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1

    tdip_f = state.transition_dipole_moment
    B = np.einsum("abc,a,bc->", epsilon, tdip_f, term1 + term2)
    return B
