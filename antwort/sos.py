import numpy as np
import itertools


def sos_static_polarizability(state):
    components = list(itertools.combinations_with_replacement([0, 1, 2], r=2))
    sos = np.zeros(len(components))
    for i, dip in enumerate(state.transition_dipole_moments):
        for c, comp in enumerate(components):
            sos[c] += 2.0 * (dip[comp[0]] * dip[comp[1]]) / state.excitation_energies[i]
    return sos


def sos_c6(state):
    points, weights = np.polynomial.legendre.leggauss(12)
    w0 = 0.3
    freqs = w0 * (1 - points) / (1 + points)
    alphas_iso = []

    for w in freqs:
        sos = sos_complex_polarizability(state, omegas=[0.0], gamma=w)[0]
        alphas_iso.append(
            1.0 / 3.0 * (sos[0].real + sos[3].real + sos[5].real)
        )
    alphas_iso = np.array(alphas_iso)
    derivative = w0 * 2 / (1 + points)**2
    integral = np.sum(alphas_iso * alphas_iso * weights * derivative)
    c6 = 3 * integral / np.pi
    return c6


def sos_complex_polarizability(state, omegas=[0.0], gamma=0.01):
    components = list(itertools.combinations_with_replacement([0, 1, 2], r=2))
    sos = np.zeros((len(omegas), len(components)), dtype=np.complex)
    for i, dip in enumerate(state.transition_dipole_moments):
        for c, comp in enumerate(components):
            sos[:, c] += (
                dip[comp[0]] * dip[comp[1]] / (state.excitation_energies[i] - omegas + np.complex(0, -gamma))
                + dip[comp[1]] * dip[comp[0]] / (state.excitation_energies[i] + omegas + np.complex(0, gamma))
            )
    return sos
