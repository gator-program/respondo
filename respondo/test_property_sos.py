"""
Test response properties against SOS
"""
import unittest

import adcc
import numpy as np

from respondo import (
    complex_polarizability,
    mcd_bterm,
    real_polarizability,
    rixs,
    static_polarizability,
    tpa_resonant,
)
from respondo.sos import (
    sos_complex_polarizability,
    sos_mcd_bterm,
    sos_rixs_amplitude,
    sos_static_polarizability,
    sos_tpa_matrix_resonant,
)

from .misc import assert_allclose_signfix, expand_test_templates
from .testdata import cache
from .testdata.static_data import xyz


def run_scf(molecule, basis, backend="pyscf"):
    scfres = adcc.backends.run_hf(
        backend,
        xyz=xyz[molecule],
        basis=basis,
    )
    return scfres


solvers = [
    "conjugate_gradient",
    "cpp",
    "jacobi_diis",
]

cases_folding = []
cases_solver = []
for c in cache.cases:
    for solver in solvers:
        cases_folding.append((c, "normal"))
        cases_solver.append((c, "normal", solver))
        if "adc2" in c:
            cases_folding.append((c, "folded"))
            cases_solver.append((c, "folded", solver))


@expand_test_templates(cases_folding)
class TestResponsePropertySosReal(unittest.TestCase):
    def template_static_polarizability(self, case, fold_doubles):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]
        alpha_ref = sos_static_polarizability(mock_state)

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        alpha = static_polarizability(
            refstate, method=method, fold_doubles=fold_doubles, conv_tol=1e-8
        )
        np.testing.assert_allclose(alpha_ref, alpha, atol=1e-7)

    def template_real_polarizability(self, case, fold_doubles):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        omega = 0.05
        alpha_ref = sos_complex_polarizability(mock_state, omegas=[omega], gamma=0.0).real

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        alpha = real_polarizability(
            refstate, method=method, omega=omega, fold_doubles=fold_doubles, conv_tol=1e-8
        ).real
        np.testing.assert_allclose(alpha_ref, alpha, atol=1e-7)

    def template_tpa_resonant(self, case, fold_doubles):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)

        # TODO: cache ADC state
        # state = cache.cached_state(molecule, basis, method, n_singlets=3)
        # --> cache.cached_backend_hf() etc.
        state = adcc.run_adc(refstate, method=method, n_singlets=5, conv_tol=1e-7)

        for ee in state.excitations:
            S_ref = sos_tpa_matrix_resonant(mock_state, final_state=ee.index)
            _, S = tpa_resonant(ee, fold_doubles=fold_doubles, conv_tol=1e-8)
            # TODO: check conv_tol stuff
            assert_allclose_signfix(S_ref, S, atol=1e-4)

    def template_mcd_bterm(self, case, fold_doubles):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)

        # TODO: cache ADC state
        # state = cache.cached_state(molecule, basis, method, n_singlets=3)
        # --> cache.cached_backend_hf() etc.
        state = adcc.run_adc(refstate, method=method, n_singlets=5, conv_tol=1e-7)

        for ee in state.excitations:
            B_ref = sos_mcd_bterm(mock_state, final_state=ee.index)
            B = mcd_bterm(ee, fold_doubles=fold_doubles, conv_tol=1e-6)
            # TODO: check conv_tol stuff
            np.testing.assert_allclose(B_ref, B, atol=1e-4)


@expand_test_templates(cases_solver)
class TestResponsePropertySosComplex(unittest.TestCase):
    def template_complex_polarizability(self, case, fold_doubles, solver):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        omega = np.random.random() + 0.1
        gamma = np.random.random() + 1e-4
        alpha_ref = sos_complex_polarizability(mock_state, omegas=[omega], gamma=gamma)

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)

        alpha = complex_polarizability(
            refstate,
            method=method,
            omega=omega,
            gamma=gamma,
            solver=solver,
            fold_doubles=fold_doubles,
            conv_tol=1e-8,
        )
        np.testing.assert_allclose(alpha_ref, alpha, atol=1e-7)


@expand_test_templates(cases_solver)
class TestResponsePropertySosComplexWithExcitation(unittest.TestCase):
    def template_rixs(self, case, fold_doubles, solver):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)

        # TODO: cache ADC state
        # state = cache.cached_state(molecule, basis, method, n_singlets=3)
        # --> cache.cached_backend_hf() etc.
        state = adcc.run_adc(refstate, method=method, n_singlets=5, conv_tol=1e-7)

        omega = 15.0
        gamma = 1e-4
        for ee in state.excitations:
            F_ref = sos_rixs_amplitude(mock_state, final_state=ee.index, omega=omega, gamma=gamma)
            _, F = rixs(
                ee,
                omega=omega,
                gamma=gamma,
                conv_tol=1e-8,
                solver=solver,
                fold_doubles=fold_doubles,
            )
            # TODO: check conv_tol stuff
            assert_allclose_signfix(F_ref, F, atol=1e-6)
