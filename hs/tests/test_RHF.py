import pytest

from hf import build_molecule
from rhf import RHF
from utils import SCFConvergeError


TEST_MOLECULES = [
    pytest.param(("tests/molecules/water.xyz",-74.94208), id="water"),
    pytest.param(("tests/molecules/allene.xyz",-114.4217), id="allene"),
    pytest.param(("tests/molecules/methane.xyz",-39.72685), id="methane"),
]

@pytest.mark.parametrize("molecule",TEST_MOLECULES)
def test_E(molecule):
    m,b = build_molecule(molecule[0], "tests/basis/sto-3g.gbs")
    wfn = RHF(m,b)
    E, _ = wfn.compute_E()
    # chemical accuracy is around 2e-4 hartree
    assert abs(E - molecule[1]) < 2e-4

@pytest.fixture
def acetaldehyde():
    basis = "tests/basis/sto-3g.gbs"
    molecule = "tests/molecules/acetaldehyde.xyz"
    wfn = RHF(*build_molecule(molecule, "tests/basis/sto-3g.gbs"))
    return wfn

def test_no_DIIS(acetaldehyde):
    """Acetaldehyde fails without DIIS"""
    with pytest.raises(SCFConvergeError): acetaldehyde.compute_E(use_diis=False)

def test_DIIS(acetaldehyde):
    """Acetaldehyde does not fail with DIIS"""
    try:
        acetaldehyde.compute_E()
    except SCFConvergeError:
        pytest.fail("DIIS stopped working :(")




