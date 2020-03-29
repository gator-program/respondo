"""
Test static polarizabilities
"""

import adcc
from pyscf import gto, scf

import numpy as np

import matplotlib.pyplot as plt

from antwort import compute_complex_polarizability


mol = gto.M(
    atom="""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    """,
    unit="Bohr",
    basis="sto-3g",
)

scfres = scf.RHF(mol)
scfres.conv_tol = 1e-8
scfres.conv_tol_grad = 1e-8
scfres.kernel()

refstate = adcc.ReferenceState(scfres)

omegas = np.linspace(0.08, 0.12, 50)
gamma = 1e-4

all_pol = [
    compute_complex_polarizability(
        "adc2", refstate, omega=w, gamma=gamma
    )
    for w in omegas
]

iso_pols_imag = [
    alpha[1][0] + alpha[1][3] + alpha[1][5] for alpha in all_pol
]

plt.plot(omegas, iso_pols_imag, "o")
plt.show()
