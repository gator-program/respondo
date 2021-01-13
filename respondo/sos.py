"""
Sum-Over-States (SOS) Expressions for response functions
"""

from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.OneParticleOperator import product_trace
import numpy as np
from itertools import permutations, product


# TODO: Add latex Sum-Over-States Expressions


def sos_static_polarizability(state):
    sos = np.zeros((3, 3))
    for i, dip in enumerate(state.transition_dipole_moment):
        for A in range(3):
            for B in range(A, 3):
                sos[A, B] += (
                    2.0 * (dip[A] * dip[B]) / state.excitation_energy_uncorrected[i]
                )
                sos[B, A] = sos[A, B]
    return sos


def sos_c6(state):
    points, weights = np.polynomial.legendre.leggauss(12)
    w0 = 0.3
    freqs = w0 * (1 - points) / (1 + points)
    alphas_iso = []

    for w in freqs:
        sos = sos_complex_polarizability(state, omegas=[0.0], gamma=w)[0]
        alphas_iso.append(
            1.0 / 3.0 * (sos[0, 0].real + sos[1, 1].real + sos[2, 2].real)
        )
    alphas_iso = np.array(alphas_iso)
    derivative = w0 * 2 / (1 + points) ** 2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3 * integral / np.pi
    return c6


def sos_complex_polarizability(state, omegas=None, gamma=0.01):
    if omegas is None:
        omegas = [0.0]
    sos = np.zeros((len(omegas), 3, 3), dtype=np.complex)
    for i, dip in enumerate(state.transition_dipole_moment):
        for A in range(3):
            for B in range(A, 3):
                sos[:, A, B] += dip[A] * dip[B] / (
                    state.excitation_energy_uncorrected[i]
                    - omegas
                    + np.complex(0, -gamma)
                ) + dip[B] * dip[A] / (
                    state.excitation_energy_uncorrected[i]
                    + omegas
                    + np.complex(0, gamma)
                )
                sos[:, B, A] = sos[:, A, B]
    return sos
