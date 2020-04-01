"""
Compute static polarizabilities in adcc
"""

import adcc
from pyscf import gto, scf

import numpy as np
import pandas as pd
import itertools

from antwort import (
    compute_static_polarizability,
)

mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvdz',
    unit="Bohr"
)

scfres = scf.RHF(mol)
scfres.conv_tol = 1e-8
scfres.conv_tol_grad = 1e-8
scfres.kernel()

refstate = adcc.ReferenceState(scfres)

nstates = 3
state = adcc.adc2(scfres, n_singlets=nstates, conv_tol=1e-6)

alpha_0 = compute_static_polarizability(
    "adc2", refstate, conv_tol=1e-6, diis=True, max_error_vectors=10
)
print(alpha_0)
