"""
Test response properties against SOS
"""
import unittest
import adcc
from pyscf import gto, scf

import numpy as np

from respondo import (
    static_polarizability,
    complex_polarizability,
    # c6_dispersion_coefficient,
)

from respondo.sos import (
    sos_static_polarizability,
    # sos_c6,
    sos_complex_polarizability,
)


class TestPolarizability(unittest.TestCase):
    def test_h2o_sto3g_adc2_fulldiag(self):
        mol = gto.M(
            atom="O 0 0 0;"
            "H 0 0 1.795239827225189;"
            "H 1.693194615993441 0 -0.599043184453037",
            basis="sto-3g",
            unit="Bohr",
        )
        scfres = scf.RHF(mol)
        scfres.conv_tol = 1e-8
        scfres.conv_tol_grad = 1e-8
        scfres.kernel()
        nstates = 65
        state = adcc.adc2(scfres, n_singlets=nstates, conv_tol=1e-6)

        refstate = adcc.ReferenceState(scfres)

        omega = np.random.random() + 0.1
        gamma = np.random.random() + 1e-4
        # SOS
        alpha_0_sos = sos_static_polarizability(state)
        # c6_sos = sos_c6(state)
        alpha_c_sos = sos_complex_polarizability(state, omegas=[omega], gamma=gamma)

        # Response Solvers
        alpha_c = complex_polarizability(
            "adc2", refstate, omega=omega, gamma=gamma, conv_tol=1e-8
        )
        # c6 = c6_dispersion_coefficient("adc2", refstate, conv_tol=1e-8)
        alpha_0 = static_polarizability("adc2", refstate, conv_tol=1e-8)

        np.testing.assert_allclose(alpha_c_sos[0], alpha_c, atol=1e-7)
        np.testing.assert_allclose(alpha_0_sos, alpha_0, atol=1e-7)
        # np.testing.assert_allclose(c6_sos, c6, atol=1e-7)
