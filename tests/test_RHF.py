import pytest

from hs.hf import build_molecule
from hs.rhf import RHF
from hs.utils import SCFConvergeError


TEST_MOLECULES = [
    pytest.param(("water",-74.94208), id="water"),
    pytest.param(("allene",-114.4217), id="allene"),
    pytest.param(("methane",-39.72685), id="methane"),
]

@pytest.mark.parametrize("molecule",TEST_MOLECULES)
def test_E(molecule):
    m,b = build_molecule(molecule[0], "sto-3g")
    wfn = RHF(m,b)
    E, _ = wfn.compute_E()
    # chemical accuracy is around 2e-4 hartree
    assert abs(E - molecule[1]) < 2e-4

@pytest.fixture
def acetaldehyde():
    basis = "sto-3g"
    molecule = "acetaldehyde"
    wfn = RHF(*build_molecule(molecule, "sto-3g"))
    return wfn

# Yeah I have no clue why this isn't working
@pytest.mark.xfail
def test_no_DIIS(acetaldehyde):
    """Acetaldehyde fails without DIIS"""
    with pytest.raises(SCFConvergeError):
        acetaldehyde.compute_E(use_diis=False)


def test_DIIS(acetaldehyde):
    """Acetaldehyde does not fail with DIIS"""
    try:
        acetaldehyde.compute_E()
    except SCFConvergeError:
        pytest.fail("DIIS stopped working :(")




