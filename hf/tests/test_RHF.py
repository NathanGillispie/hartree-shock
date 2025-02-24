import pytest

from hf import build_molecule
from rhf import RHF
from utils import SCFConvergeError


TEST_MOLECULES = [
    pytest.param(("tests/molecules/water.xyz",-74.942079928192), id="water"),
    pytest.param(("tests/molecules/allene.xyz",-114.4221005655933538), id="allene"),
    pytest.param(("tests/molecules/methane.xyz",-39.726850324347), id="methane"),
]

@pytest.mark.parametrize("molecule",TEST_MOLECULES)
def test_E(molecule):
    m,b = build_molecule(molecule[0], "tests/basis/sto-3g.gbs")
    wfn = RHF(m,b)
    E, _ = wfn.compute_E()
    # chemical accuracy is around 2e-4 hartree
    assert abs(E - molecule[1]) < 2e-4

def test_converge():
    """Acetaldehyde fails"""
    basis = "tests/basis/sto-3g.gbs"
    molecule = "tests/molecules/acetaldehyde.xyz"
    wfn = RHF(*build_molecule(molecule, "tests/basis/sto-3g.gbs"))

    with pytest.raises(SCFConvergeError): wfn.compute_E()

