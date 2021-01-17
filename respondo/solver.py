import sys
import numpy as np
import scipy.linalg as la

from adcc import lincomb
from adcc.AmplitudeVector import AmplitudeVector

from adcc.solver.explicit_symmetrisation import IndexSymmetrisation

from .MatrixWrapper import (
    ComplexPolarizationPropagatorPinv,
    MatrixWrapper,
    ComplexPolarizationPropagatorMatrixFolded,
)
from .cpp_algebra import ResponseVector


class State:
    def __init__(self):
        self.solution = None       # Current approximation to the solution
        self.residual = None       # Current residual
        self.residual_norm = None  # Current residual norm
        self.converged = False     # Flag whether iteration is converged
        self.n_iter = 0            # Number of iterations
        self.n_applies = 0         # Number of applies
        self.n_ss_vectors = 0      # Number of subspace vectors


def default_print(state, identifier, file=sys.stdout):
    if identifier == "start" and state.n_iter == 0:
        print("Niter residual_norm    n_vec", file=file)
    elif identifier == "next_iter":
        fmt = "{n_iter:3d}  {residual:12.5g}    {n_vec:3d}"
        print(fmt.format(n_iter=state.n_iter,
                         residual=np.max(state.residual_norm),
                         n_vec=state.n_ss_vectors), file=file)
    elif identifier == "is_converged":
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies)


def test_list_ortho(vectors, label=""):
    print(f"Number of vectors = {len(vectors)}")
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i == j:
                ovlp = 1 - vectors[i] @ vectors[j]
                if ovlp > 1e-14:
                    print(label, i, j, "WARNING: ", ovlp)
            elif i < j:
                ovlp = vectors[i] @ vectors[j]
                if ovlp > 1e-14:
                    print(label, i, j, "WARNING: ", ovlp)


def modified_gram_schmidt(guesses):
    """
    Modified Gram-Schmidt to orthonormalize @guesses
    """
    for k in range(len(guesses)):
        # Project out the components of the current subspace
        guesses[k] = guesses[k] / np.sqrt(guesses[k] @ guesses[k])
        for ll in range(k + 1, len(guesses)):
            guesses[ll] = guesses[ll] - (guesses[k] @ guesses[ll]) * guesses[k]


def filter_by_overlap(subspace, min_norm, max_add=None):
    n_vec = len(subspace)
    singles_subspace = [AmplitudeVector(s.ph) for s in subspace]
    for w in range(n_vec):
        norm = np.sqrt(singles_subspace[w] @ singles_subspace[w])
        singles_subspace[w] = singles_subspace[w] / norm

    ovlp_mat = np.zeros((n_vec, n_vec))
    for i in range(n_vec):
        for j in range(n_vec):
            if i <= j:
                ovlp_mat[i, j] = singles_subspace[i] @ singles_subspace[j]
                ovlp_mat[j, i] = ovlp_mat[i, j]

    evals, evecs = la.eigh(ovlp_mat)
    mask = np.where(np.abs(evals) > min_norm)[0]
    new_subspace = []
    if max_add:
        for m in mask[:max_add]:
            new_subspace.append(lincomb(evecs[m], subspace, evaluate=True))
    else:
        for m in mask:
            new_subspace.append(lincomb(evecs[m], subspace, evaluate=True))
    return new_subspace


def cpp_solver(matrix, rhs, x0, omega, gamma, conv_tol=1e-9,
               explicit_symmetrisation=IndexSymmetrisation,
               residual_min_norm=None,
               Pinv=None,
               max_subspace=None, max_iter=100, callback=default_print):
    if callback is None:
        def callback(state, identifier):
            pass

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if Pinv is None:
        Pinv = ComplexPolarizationPropagatorPinv(
            matrix.diagonal(), omega, gamma
        )

    def is_converged(state):
        state.converged = state.residual_norm < conv_tol
        return state.converged

    state = State()

    if max_subspace is None:
        max_subspace = 1000

    if isinstance(x0, ResponseVector):
        real = explicit_symmetrisation.symmetrise(x0.real)
        imag = explicit_symmetrisation.symmetrise(x0.imag)
        x0 = [real, imag]
    elif isinstance(x0, AmplitudeVector):
        x0 = [x0]
    else:
        raise NotImplementedError()

    # x0 = filter_by_overlap(x0, 1e-12)
    modified_gram_schmidt(x0)
    # test_list_ortho(x0)

    SS = x0

    Mss_cont = np.empty((max_subspace, max_subspace))
    Ax_cont = []
    rhs_SS_cont = np.empty(max_subspace * 2)

    n_problem = matrix.shape[1]
    n_block = len(SS)
    n_ss_vec = n_block
    state.n_ss_vectors = n_ss_vec

    eps = np.finfo(float).eps
    if residual_min_norm is None:
        residual_min_norm = 2 * n_problem * eps

    callback(state, "start")
    while state.n_iter < max_iter:
        state.n_iter += 1
        # if added:
        #     residual_min_norm = 1e-6
        AxBlock = []
        if n_block > 0:
            state.n_applies += n_block
            AxBlock = matrix @ SS[-n_block:]

        Ax_cont.extend(AxBlock)
        Mss = Mss_cont[:n_ss_vec, :n_ss_vec]

        for i in range(n_ss_vec):
            for j in range(n_ss_vec - n_block, n_ss_vec):
                Mss[i, j] = SS[i] @ Ax_cont[j]
                Mss[j, i] = Mss[i, j]

        gamma_mat = gamma * np.eye(Mss.shape[0])
        full_Mss = np.block([
            [Mss, gamma_mat],
            [gamma_mat, -Mss]
        ])

        # TODO: only works correctly for purely real rhs
        rhs_SS = rhs_SS_cont[:n_ss_vec]
        rhs_SS[-n_block:] = rhs.real @ SS[-n_block:]
        rhs_SS = np.append(rhs_SS, np.zeros_like(rhs_SS))

        omega_SS = omega * np.eye(n_ss_vec)
        omega_zeros = np.zeros_like(omega_SS)
        omega_SS = np.block([
            [omega_SS, omega_zeros],
            [omega_zeros, -omega_SS]
        ])

        # solve the subspace equation
        Msso = full_Mss - omega_SS
        x = la.solve(Msso, rhs_SS)

        # Compute residuals
        Axfull_real = lincomb(x[:n_ss_vec], Ax_cont, evaluate=True)
        Axfull_imag = lincomb(x[n_ss_vec:], Ax_cont, evaluate=True)
        asym_tmp_real = lincomb(
            x[:n_ss_vec] * omega, SS, evaluate=True
        )
        asym_tmp_imag = lincomb(
            x[n_ss_vec:] * omega, SS, evaluate=True
        )
        gamma_real = lincomb(x[n_ss_vec:] * gamma, SS, evaluate=True)
        gamma_imag = lincomb(x[:n_ss_vec] * gamma, SS, evaluate=True)
        residual_real = (
            Axfull_real - asym_tmp_real - rhs.real + gamma_real
        )
        residual_imag = (
            -1.0 * Axfull_imag + asym_tmp_imag + rhs.imag + gamma_imag
        )

        residual = ResponseVector(residual_real, residual_imag)
        state.residual_norm = np.sqrt(residual @ residual)

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            real_part = lincomb(
                np.transpose(x[:n_ss_vec]), SS, evaluate=True
            )
            imag_part = lincomb(
                np.transpose(x[n_ss_vec:]), SS, evaluate=True
            )
            state.solution = ResponseVector(real_part, imag_part)
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations "
                                 f"(== {max_iter}) reached in cpp_solver.")

        if Pinv:
            precond = Pinv @ residual
        else:
            precond = residual
        real = explicit_symmetrisation.symmetrise(precond.real)
        imag = explicit_symmetrisation.symmetrise(precond.imag)
        preconds = [real, imag]

        # if not (len(preconds) <= 2 or gamma == 0.0):
        #     preconds = filter_by_overlap(preconds, residual_min_norm)
        modified_gram_schmidt(preconds)
        n_ss_added = 0
        for pvec in preconds:
            pvec = pvec / np.sqrt(pvec @ pvec)
            pvec = pvec - lincomb(pvec @ SS, SS, evaluate=True)
            pnorm = np.sqrt(pvec @ pvec)
            if pnorm > residual_min_norm:
                SS.append(pvec / pnorm)
                n_ss_added += 1
                n_ss_vec = len(SS)
                state.n_ss_vectors = n_ss_vec

        n_block = n_ss_added
        # if n_ss_added > 0:
        #     added = True
        # else:
        #     residual_min_norm *= 0.1
        #     added = False

        if n_ss_vec >= max_subspace:
            real_part = lincomb(
                np.transpose(x[:n_ss_vec - n_ss_added]),
                SS[:-n_ss_added], evaluate=True
            )
            imag_part = lincomb(
                np.transpose(x[n_ss_vec - n_ss_added:]),
                SS[:-n_ss_added], evaluate=True
            )
            SS = [real_part, imag_part]
            # SS = filter_by_overlap(
            #     SS, residual_min_norm / 1000
            # )
            modified_gram_schmidt(SS)
            Ax_cont = []
            n_block = len(SS)
            n_ss_vec = len(SS)
            state.n_ss_vec = n_ss_vec


def cpp_solver_folded(wrapper, rhs, x0, omega, gamma, conv_tol=1e-9,
                      residual_min_norm=None,
                      max_subspace=None, max_iter=100, callback=default_print):
    if not isinstance(wrapper, MatrixWrapper):
        raise TypeError()
    matrix_folded = wrapper._wrapped
    if not isinstance(matrix_folded,
                      ComplexPolarizationPropagatorMatrixFolded):
        raise TypeError()

    if callback is None:
        def callback(state, identifier):
            pass

    Pinv = wrapper.preconditioner

    def is_converged(state):
        state.converged = state.residual_norm < conv_tol
        return state.converged

    state = State()

    if max_subspace is None:
        max_subspace = 1000

    if isinstance(x0, ResponseVector):
        x0 = [x0.real, x0.imag]
    elif isinstance(x0, AmplitudeVector):
        x0 = [x0]
    else:
        raise NotImplementedError()

    # x0 = filter_by_overlap(x0, 1e-12)
    modified_gram_schmidt(x0)
    # test_list_ortho(x0)

    SS = x0

    Mss_cont = np.empty((max_subspace, max_subspace))
    Gss_cont = np.empty((max_subspace, max_subspace))
    Ax_cont = []
    Dx_cont = []
    Gx_cont = []
    rhs_SS_real_cont = np.empty(max_subspace)
    rhs_SS_imag_cont = np.empty(max_subspace)

    n_problem = matrix_folded.shape[1]
    n_block = len(SS)
    n_ss_vec = n_block
    state.n_ss_vectors = n_ss_vec

    eps = np.finfo(float).eps
    if residual_min_norm is None:
        residual_min_norm = 2 * n_problem * eps

    callback(state, "start")
    while state.n_iter < max_iter:
        state.n_iter += 1
        # if added:
        #     residual_min_norm = 1e-6
        AxBlock = []
        if n_block > 0:
            state.n_applies += n_block
            AxBlock = [
                AmplitudeVector(ph=wrapper.matrix.block_apply('ph_ph', sv.ph))
                for sv in SS[-n_block:]
            ]
        Ax_cont.extend(AxBlock)

        for i in range(n_block):
            Dx, Gx = matrix_folded._apply_D_G(SS[-n_block:][i])
            Dx_cont.append(AxBlock[i] - Dx)
            Gx_cont.append(Gx)

        Mss = Mss_cont[:n_ss_vec, :n_ss_vec]
        Gss = Gss_cont[:n_ss_vec, :n_ss_vec]
        for i in range(n_ss_vec):
            for j in range(n_ss_vec - n_block, n_ss_vec):
                Mss[i, j] = SS[i] @ Dx_cont[j]
                Mss[j, i] = Mss[i, j]
                Gss[i, j] = SS[i] @ Gx_cont[j]
                Gss[j, i] = Gss[i, j]

        full_Mss = np.block([
            [Mss, Gss],
            [Gss, -Mss]
        ])

        rhs_SS_real = rhs_SS_real_cont[:n_ss_vec]
        rhs_SS_imag = rhs_SS_imag_cont[:n_ss_vec]

        rhs_SS_real[-n_block:] = rhs.real @ SS[-n_block:]
        rhs_SS_imag[-n_block:] = rhs.imag @ SS[-n_block:]
        rhs_SS = np.append(rhs_SS_real, rhs_SS_imag)

        x = la.solve(full_Mss, rhs_SS)

        Axfull_real = lincomb(x[:n_ss_vec], Dx_cont, evaluate=True)
        Axfull_imag = lincomb(x[n_ss_vec:], Dx_cont, evaluate=True)

        gamma_real = lincomb(x[n_ss_vec:], Gx_cont, evaluate=True)
        gamma_imag = lincomb(x[:n_ss_vec], Gx_cont, evaluate=True)

        residual_real = (Axfull_real - rhs.real + gamma_real)
        residual_imag = (-1.0 * Axfull_imag - rhs.imag + gamma_imag)

        residual = ResponseVector(residual_real, residual_imag)
        state.residual_norm = np.sqrt(residual @ residual)

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            real_part = lincomb(
                np.transpose(x[:n_ss_vec]), SS, evaluate=True
            )
            imag_part = lincomb(
                np.transpose(x[n_ss_vec:]), SS, evaluate=True
            )
            state.solution = ResponseVector(real_part, imag_part)
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations "
                                 f"(== {max_iter}) reached in cpp_solver.")

        if Pinv:
            precond = Pinv @ residual
        else:
            precond = residual
        preconds = [precond.real, precond.imag]

        # if not (len(preconds) <= 2 or gamma == 0.0):
        #     preconds = filter_by_overlap(preconds, residual_min_norm)
        modified_gram_schmidt(preconds)
        n_ss_added = 0
        for pvec in preconds:
            pvec = pvec / np.sqrt(pvec @ pvec)
            pvec = pvec - lincomb(pvec @ SS, SS, evaluate=True)
            pnorm = np.sqrt(pvec @ pvec)
            if pnorm > residual_min_norm:
                SS.append(pvec / pnorm)
                n_ss_added += 1
                n_ss_vec = len(SS)
                state.n_ss_vectors = n_ss_vec

        n_block = n_ss_added
        # if n_ss_added > 0:
        #     added = True
        # else:
        #     residual_min_norm *= 0.1
        #     added = False

        if n_ss_vec >= max_subspace:
            real_part = lincomb(
                np.transpose(x[:n_ss_vec - n_ss_added]),
                SS[:-n_ss_added], evaluate=True
            )
            imag_part = lincomb(
                np.transpose(x[n_ss_vec - n_ss_added:]),
                SS[:-n_ss_added], evaluate=True
            )
            SS = [real_part, imag_part]
            # SS = filter_by_overlap(
            #     SS, residual_min_norm / 1000
            # )
            modified_gram_schmidt(SS)
            Ax_cont = []
            Dx_cont = []
            Gx_cont = []
            n_block = len(SS)
            n_ss_vec = len(SS)
            state.n_ss_vec = n_ss_vec
