"""
Test the the fock matrix is orthogonal in the MO basis
"""
import pytest
import numpy as np

from hs.hf import build_molecule
from hs.rhf import RHF

TOLERANCE = 1e-7

# SLOWWW
TEST_MOLECULES = [
    "water",
    "allene",
    "methane"
]

TEST_BASIS = [
    "sto-3g",
]

@pytest.mark.parametrize("molecule", TEST_MOLECULES)
@pytest.mark.parametrize("basis", TEST_BASIS)
def test_fock(molecule, basis):

    wfn = RHF(*build_molecule(molecule, basis))
    _, _ = wfn.compute_E()

    F_MO = np.einsum("mj,ni,mn->ij",wfn.C,wfn.C,wfn.F_AO, optimize=True)
    assert np.abs(F_MO-F_MO.T).sum() < TOLERANCE
