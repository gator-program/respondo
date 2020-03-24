import itertools

from adcc import AdcMatrix, LazyMp, ReferenceState

import numpy as np

from adcc.modified_transition_moments import compute_modified_transition_moments

from adcc.solver.conjugate_gradient import default_print
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver import IndexSymmetrisation

from .utils import jacobi

from .cpp_algebra import (ResponseVector,
                          ComplexPolarizationPropagatorPinv,
                          conjugate_gradient,
                          ResponseVectorSymmetrisation)
from .cpp_algebra import ComplexPolarizationPropagatorMatrix as CppMatrix


_comps = ['x', 'y', 'z']


def compute_static_polarizability(matrix_method, scfres, **solver_args):
    """
    Compute the static polarizability of the electronic
    ground state. Tensor is returned in alphabetical ordering
    (xx, xy, xz, yy, yz, zz)
    """
    reference_state = ReferenceState(scfres)
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
        res = jacobi(
            matrix, rhs=rhs, x0=x0, callback=default_print,
            explicit_symmetrisation=explicit_symmetrisation,
            **solver_args
        )
        solutions.append(res.solution)

    # xx, xy, xz, yy, yz, zz
    components = list(itertools.combinations_with_replacement([0, 1, 2], r=2))
    polarizability = np.zeros(len(components))
    for c, comp in enumerate(components):
        polarizability[c] = 2.0 * solutions[comp[1]] @ rhss[comp[0]]
    # TODO: return as numpy array with all elements
    return polarizability


def compute_complex_polarizability(matrix_method, scfres, omega=0.0, gamma=0.0,
                                   solver=conjugate_gradient,
                                   **solver_args):
    """
    Compute the complex frequency-dependent polarizability of the electronic
    ground state. Tensor is returned in alphabetical ordering
    (xx, xy, xz, yy, yz, zz)
    """
    # TODO: allow for multiple frequencies from outside, multi-frequency solver
    reference_state = ReferenceState(scfres)
    dips = reference_state.operators.electric_dipole
    ground_state = LazyMp(reference_state)
    matrix = AdcMatrix(matrix_method, ground_state)
    rhss = [
        compute_modified_transition_moments(matrix, dip, "adc2")
        for dip in dips
    ]
    cpp_matrix = CppMatrix(matrix, gamma=gamma, omega=omega)
    Pinv = ComplexPolarizationPropagatorPinv(cpp_matrix, shift=omega, gamma=gamma)
    solutions_cg = []
    for mu in range(3):
        rhs = ResponseVector(rhss[mu])
        x0 = Pinv @ rhs
        rsymm = ResponseVectorSymmetrisation(matrix)
        guess_symm = rsymm.symmetrise(x0, rhs)

        cpp_matrix.omega = omega
        Pinv.shift = omega
        print(f"Solving complex response equation for component {_comps[mu]}.")
        res1 = solver(
            cpp_matrix, rhs=rhs, x0=guess_symm,
            callback=default_print, Pinv=Pinv,
            explicit_symmetrisation=rsymm,
            **solver_args
        )
        cpp_matrix.omega = -omega
        Pinv.shift = -omega
        res2 = solver(
            cpp_matrix, rhs=rhs, x0=guess_symm,
            callback=default_print, Pinv=Pinv,
            explicit_symmetrisation=rsymm,
            **solver_args
        )
        cpp_matrix.omega = omega
        Pinv.shift = omega
        solutions_cg.append((res1.solution, res2.solution))

    # xx, xy, xz, yy, yz, zz
    components = list(itertools.combinations_with_replacement([0, 1, 2], r=2))

    polarizability_real_cg = np.zeros(len(components))
    polarizability_imag_cg = np.zeros(len(components))
    for c, comp in enumerate(components):
        sol1 = solutions_cg[comp[1]][0]
        sol2 = solutions_cg[comp[1]][1]
        polarizability_real_cg[c] = sol1.real @ rhss[comp[0]] + sol2.real @ rhss[comp[0]]
        polarizability_imag_cg[c] = sol1.imag @ rhss[comp[0]] - sol2.imag @ rhss[comp[0]]
    # TODO: return as complex numpy array
    return (polarizability_real_cg, polarizability_imag_cg)


def compute_c6_dispersion_coefficient(matrix_method, scfres, **solver_args):
    """
    Compute the ground state C6 dispersion coefficient by quadrature
    """

    points, weights = np.polynomial.legendre.leggauss(12)
    w0 = 0.3
    freqs = w0 * (1 - points) / (1 + points)
    alphas_iso = []

    for w in freqs:
        re, im = compute_complex_polarizability(
            matrix_method, scfres, omega=0.0, gamma=w, **solver_args
        )
        alphas_iso.append(
            1.0 / 3.0 * (re[0] + re[3] + re[5])
        )
    alphas_iso = np.array(alphas_iso)
    derivative = w0 * 2 / (1 + points)**2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3.0 * integral / np.pi
    return c6
