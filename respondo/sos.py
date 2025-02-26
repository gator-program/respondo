"""
Sum-Over-States (SOS) Expressions for response functions
"""

import numpy as np

# TODO: Add latex Sum-Over-States Expressions


def sos_static_polarizability(state):
    sos = np.zeros((3, 3))
    for i, dip in enumerate(state.transition_dipole_moment):
        for A in range(3):
            for B in range(A, 3):
                sos[A, B] += 2.0 * (dip[A] * dip[B]) / state.excitation_energy_uncorrected[i]
                sos[B, A] = sos[A, B]
    return sos


def sos_c6(state):
    points, weights = np.polynomial.legendre.leggauss(12)
    w0 = 0.3
    freqs = w0 * (1 - points) / (1 + points)
    alphas_iso = []

    for w in freqs:
        sos = sos_complex_polarizability(state, omegas=[0.0], gamma=w)[0]
        alphas_iso.append(1.0 / 3.0 * (sos[0, 0].real + sos[1, 1].real + sos[2, 2].real))
    alphas_iso = np.array(alphas_iso)
    derivative = w0 * 2 / (1 + points) ** 2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3 * integral / np.pi
    return c6


def sos_complex_polarizability(state, omegas=None, gamma=0.01):
    if omegas is None:
        omegas = [0.0]
    sos = np.zeros((len(omegas), 3, 3), dtype=complex)
    for i, dip in enumerate(state.transition_dipole_moment):
        for A in range(3):
            for B in range(A, 3):
                sos[:, A, B] += dip[A] * dip[B] / (
                    state.excitation_energy_uncorrected[i] - omegas + complex(0, -gamma)
                ) + dip[B] * dip[A] / (
                    state.excitation_energy_uncorrected[i] + omegas + complex(0, gamma)
                )
                sos[:, B, A] = sos[:, A, B]
    return np.squeeze(sos)


def sos_rixs_amplitude(state, final_state=0, omega=0.0, gamma=0.01):
    """
    SOS for RIXS amplitude in the rotating wave approximation
    """
    F = np.zeros((3, 3), dtype=complex)
    s2s_tdm = state.transition_dipole_moment_s2s
    for ee in range(state.excitation_energy.size):
        tdm_fn = s2s_tdm[ee, final_state]
        for A in range(3):
            for B in range(3):
                F[A, B] += (
                    tdm_fn[A]
                    * state.transition_dipole_moment[ee][B]
                    / (state.excitation_energy_uncorrected[ee] - omega - complex(0, gamma))
                )
    # ground state coupling
    tdip_f = state.transition_dipole_moment[final_state]
    pm = state.property_method.replace("adc", "")
    gs_dip_moment = state.ground_state.dipole_moment[pm]
    for A in range(3):
        for B in range(3):
            F[A, B] += (tdip_f[A] * gs_dip_moment[B]) / (-omega - complex(0, gamma)) - (
                gs_dip_moment[A] * tdip_f[B]
            ) / (omega + complex(0, gamma) - state.excitation_energy_uncorrected[final_state])
    return F


def sos_tpa_matrix_resonant(state, final_state=0):
    ret = np.zeros((3, 3))
    w = state.excitation_energy_uncorrected[final_state] / 2.0
    nstates = state.excitation_energy.size
    s2s_tdm = state.transition_dipole_moment_s2s
    for k in range(nstates):
        for A in range(3):
            for B in range(A, 3):
                ret[A, B] += (
                    state.transition_dipole_moment[k][A]
                    * s2s_tdm[k, final_state, B]
                    / (state.excitation_energy_uncorrected[k] - w)
                )
                ret[A, B] += (
                    state.transition_dipole_moment[k][B]
                    * s2s_tdm[k, final_state, A]
                    / (state.excitation_energy_uncorrected[k] - w)
                )
                ret[B, A] = ret[A, B]
    return ret


def sos_mcd_bterm(state, final_state=0):
    term1 = np.zeros((3, 3))
    term2 = np.zeros_like(term1)

    e_f = state.excitation_energy_uncorrected[final_state]
    tdip_f = state.transition_dipole_moment[final_state]

    s2s_tdm = state.transition_dipole_moment_s2s
    s2s_tdm_mag = state.transition_magnetic_moment_s2s
    nstates = state.excitation_energy.size
    for k in range(nstates):
        for A in range(3):
            for B in range(3):
                term1[A, B] -= (
                    state.transition_magnetic_dipole_moment[k][B]
                    * s2s_tdm[final_state, k, A]
                    / (state.excitation_energy_uncorrected[k])
                )
                if k != final_state:
                    term2[A, B] += (
                        state.transition_dipole_moment[k][A]
                        * s2s_tdm_mag[final_state, k, B]
                        / (state.excitation_energy_uncorrected[k] - e_f)
                    )

    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1
    B = -1.0 * np.einsum("abc,a,bc->", epsilon, tdip_f, term1 + term2)
    return B
