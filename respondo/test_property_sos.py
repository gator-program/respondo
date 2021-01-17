"""
Test response properties against SOS
"""
import unittest
import adcc

import numpy as np

from .testdata.static_data import xyz
from .testdata import cache
from .misc import expand_test_templates

from respondo import (
    static_polarizability,
    real_polarizability,
    complex_polarizability,
)

from respondo.sos import (
    sos_static_polarizability,
    sos_complex_polarizability,
)


def run_scf(molecule, basis, backend="pyscf"):
    scfres = adcc.backends.run_hf(
        backend, xyz=xyz[molecule],
        basis=basis,
    )
    return scfres


solvers = [
    'conjugate_gradient',
    'cpp'
]

cases_folding = []
cases_solver = []
for c in cache.cases:
    for solver in solvers:
        cases_folding.append((c, 'normal'))
        cases_solver.append((c, 'normal', solver))
        if "adc2" in c:
            cases_folding.append((c, 'folded'))
            cases_solver.append((c, 'folded', solver))


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
            refstate, method=method, fold_doubles=fold_doubles,
            conv_tol=1e-8,
        )
        np.testing.assert_allclose(alpha_ref, alpha, atol=1e-7)

    def template_real_polarizability(self, case, fold_doubles):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        omega = 0.05
        alpha_ref = sos_complex_polarizability(
            mock_state, omegas=[omega], gamma=0.0
        ).real

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)
        alpha = real_polarizability(
            refstate, method=method, omega=omega,
            fold_doubles=fold_doubles, conv_tol=1e-8
        ).real
        np.testing.assert_allclose(alpha_ref, alpha, atol=1e-7)


@expand_test_templates(cases_solver)
class TestResponsePropertySosComplex(unittest.TestCase):
    def template_complex_polarizability(self, case, fold_doubles, solver):
        fold_doubles = fold_doubles == "folded"
        molecule, basis, method = case.split("_")
        mock_state = cache.data_fulldiag[case]

        omega = np.random.random() + 0.1
        gamma = np.random.random() + 1e-4
        alpha_ref = sos_complex_polarizability(
            mock_state, omegas=[omega], gamma=gamma
        )

        scfres = run_scf(molecule, basis)
        refstate = adcc.ReferenceState(scfres)

        alpha = complex_polarizability(
            refstate, method=method, omega=omega, gamma=gamma,
            solver=solver,
            fold_doubles=fold_doubles, conv_tol=1e-8
        )
        np.testing.assert_allclose(alpha_ref, alpha, atol=1e-7)
