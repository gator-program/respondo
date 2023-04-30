"""
Test static polarizabilities
"""

import adcc
from pyscf import gto, scf

import numpy as np


from respondo import complex_polarizability
from respondo.polarizability import one_photon_absorption_cross_section
from respondo import c6_dispersion_coefficient

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
matrix = adcc.AdcMatrix('adc2', refstate)

omegas = np.linspace(0.59, 0.61, 30)
gamma = 1e-3

c6 = c6_dispersion_coefficient(matrix, 'adc2')

all_pol = [
    complex_polarizability(
        matrix, omega=w, gamma=gamma, conv_tol=1e-3, fold_doubles=True
    )
    for w in omegas
]
all_pol = np.array(all_pol)
sigmas = one_photon_absorption_cross_section(all_pol, omegas)

try:
    import matplotlib.pyplot as plt
    plt.plot(omegas, sigmas, "o")
    plt.xlabel("frequency [a.u.]")
    plt.ylabel(r"$\sigma(\omega)$ [a.u.]")
    plt.show()
except ModuleNotFoundError:
    print("Please install matplotlib to plot the results.")
