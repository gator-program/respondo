import numpy as np

from adcc.solver.conjugate_gradient import conjugate_gradient, default_print
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace

from .MatrixWrapper import MatrixWrapper


def solve_response(matrix, rhs, omega, gamma, solver=conjugate_gradient,
                   fold_doubles=False, **solver_args):

    wrapper = MatrixWrapper(matrix, omega, gamma, fold_doubles=fold_doubles)
    rhs_processed = wrapper.form_rhs(rhs)
    x0 = wrapper.preconditioner @ rhs_processed
    # solve system of linear equations
    res = solver(
        wrapper,
        rhs=rhs_processed,
        x0=x0,
        callback=default_print,
        Pinv=wrapper.preconditioner,
        explicit_symmetrisation=wrapper.explicit_symmetrisation,
        **solver_args,
    )
    assert res.converged
    print(wrapper)
    solution = wrapper.form_solution(res.solution, rhs)
    return solution


# from_vecs * B(ops) * to_vecs
def transition_polarizability(method, ground_state, from_vecs, ops, to_vecs):
    if not isinstance(from_vecs, list):
        from_vecs = [from_vecs]
    if not isinstance(to_vecs, list):
        to_vecs = [to_vecs]
    if not isinstance(ops, list):
        ops = [ops]

    ret = np.zeros((len(from_vecs), len(ops), len(to_vecs)))
    for i, from_vec in enumerate(from_vecs):
        for j, to_vec in enumerate(to_vecs):
            tdm = state2state_transition_dm(
                method, ground_state, from_vec, to_vec
            )
            for k, op in enumerate(ops):
                ret[i, k, j] = product_trace(tdm, op)
    return np.squeeze(ret)


def transition_polarizability_complex(method, ground_state,
                                      from_vecs, ops, to_vecs):
    if not isinstance(from_vecs, list):
        from_vecs = [from_vecs]
    if not isinstance(to_vecs, list):
        to_vecs = [to_vecs]
    if not isinstance(ops, list):
        ops = [ops]

    # TODO: maybe use non-complex transition_polarizability function?
    # TODO: "recognize" ResponseVector/AmplitudeVector
    ret = np.zeros((len(from_vecs), len(ops), len(to_vecs)), dtype=np.complex)
    for i, from_vec in enumerate(from_vecs):
        for j, to_vec in enumerate(to_vecs):
            # TODO: optimize performance...
            # compute dot product?
            tdm_real_real = state2state_transition_dm(
                method, ground_state, from_vec.real, to_vec.real
            )
            tdm_imag_real = state2state_transition_dm(
                method, ground_state, from_vec.imag, to_vec.real
            )
            tdm_real_imag = state2state_transition_dm(
                method, ground_state, from_vec.real, to_vec.imag
            )
            tdm_imag_imag = state2state_transition_dm(
                method, ground_state, from_vec.imag, to_vec.imag
            )
            real = tdm_real_real - tdm_imag_imag
            imag = tdm_imag_real + tdm_real_imag
            for k, op in enumerate(ops):
                ret[i, k, j] = np.complex(
                    product_trace(real, op), product_trace(imag, op)
                )
    return np.squeeze(ret)
