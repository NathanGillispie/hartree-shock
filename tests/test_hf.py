import pytest

# from hs import build_molecule
# from ints import integrals, nuclear_repulsion
from hs.hf import build_molecule
from hs.ints import integrals, nuclear_repulsion

TEST_MOLECULES = [
    pytest.param("water", id="water"),
    pytest.param("acetaldehyde", id="acetaldehyde"),
    pytest.param("NiH2", id="NiH2"),
]

TEST_BASIS = [
    pytest.param("6-31G", id="6-31G"),
    pytest.param("def2-TZVP", id="def2-TZVP"),
]

@pytest.mark.parametrize("molecule", TEST_MOLECULES)
@pytest.mark.parametrize("basis", TEST_BASIS)
def test_build_molecule(molecule, basis):
    build_molecule(molecule, basis)

@pytest.mark.parametrize("molecule", TEST_MOLECULES)
@pytest.mark.parametrize("basis", TEST_BASIS)
def test_integrals(molecule, basis):
    m, b = build_molecule(molecule, basis)
    mints =integrals(m,b,True)

    ovlp = mints.overlap()
    E_nuc =nuclear_repulsion(m)
    kinetic = mints.kinetic_energy()
    nuc_attr = mints.nuclear_attraction()
    # Takes too long
    #eri = mints.electron_repulsion()

