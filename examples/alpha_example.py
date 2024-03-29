"""
Compute static polarizabilities in adcc
"""

import adcc
from pyscf import gto, scf

from respondo import static_polarizability

mol = gto.M(
    atom="O 0 0 0;"
    "H 0 0 1.795239827225189;"
    "H 1.693194615993441 0 -0.599043184453037",
    basis="cc-pvdz",
    unit="Bohr",
)

scfres = scf.RHF(mol)
scfres.conv_tol = 1e-8
scfres.conv_tol_grad = 1e-8
scfres.kernel()

refstate = adcc.ReferenceState(scfres)

alpha_0 = static_polarizability(refstate, 'adc2', conv_tol=1e-6)
print(alpha_0)
