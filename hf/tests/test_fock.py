"""
Test the the fock matrix is orthogonal in the MO basis
"""
import pytest
import numpy as np

from hf import build_molecule
from rhf import RHF

TOLERANCE = 1e-7

# SLOWWW
TEST_MOLECULES = [
    pytest.param("tests/molecules/water.xyz", id="water"),
    pytest.param("tests/molecules/allene.xyz", id="allene"),
    pytest.param("tests/molecules/methane.xyz", id="methane"),
]

TEST_BASIS = [
    pytest.param("tests/basis/sto-3g.gbs", id="sto-3g"),
    #pytest.param("tests/basis/cc-pVDZ.gbs", id="cc-pVDZ"),
]

@pytest.mark.parametrize("molecule", TEST_MOLECULES)
@pytest.mark.parametrize("basis", TEST_BASIS)
def test_fock(molecule, basis):

    wfn = RHF(*build_molecule(molecule, basis))
    _, _ = wfn.compute_E()

    F_MO = np.einsum("mj,ni,mn->ij",wfn.C,wfn.C,wfn.F_AO, optimize=True)
    assert np.abs(F_MO-F_MO.T).sum() < TOLERANCE
