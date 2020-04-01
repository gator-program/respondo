import numpy as np
import itertools


def sos_static_polarizability(state):
    sos = np.zeros((3, 3))
    for i, dip in enumerate(state.transition_dipole_moments):
        for c1 in range(3):
            for c2 in range(c1, 3):
                sos[c1, c2] += 2.0 * (dip[c1] * dip[c2]) / state.excitation_energies[i]
                sos[c2, c1] = sos[c1, c2]
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
    derivative = w0 * 2 / (1 + points)**2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3 * integral / np.pi
    return c6


def sos_complex_polarizability(state, omegas=None, gamma=0.01):
    if omegas is None:
        omegas = [0.0]
    sos = np.zeros((len(omegas), 3, 3), dtype=np.complex)
    for i, dip in enumerate(state.transition_dipole_moments):
        for c1 in range(3):
            for c2 in range(c1, 3):
                sos[:, c1, c2] += (
                    dip[c1] * dip[c2] / (state.excitation_energies[i] - omegas + np.complex(0, -gamma))
                    + dip[c2] * dip[c1] / (state.excitation_energies[i] + omegas + np.complex(0, gamma))
                )
                sos[:, c2, c1] = sos[:, c1, c2]
    return sos
