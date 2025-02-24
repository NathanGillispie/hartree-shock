import pytest

from hf import build_molecule
from ints import integrals, nuclear_repulsion

TEST_MOLECULES = [
    pytest.param("tests/molecules/water.xyz", id="water"),
    pytest.param("tests/molecules/acetaldehyde.xyz", id="acetaldehyde"),
    pytest.param("tests/molecules/allene.xyz", id="allene"),
    pytest.param("tests/molecules/benzene.xyz", id="benzene"),
]

TEST_BASIS = [
    pytest.param("tests/basis/sto-3g.gbs", id="sto-3g"),
    pytest.param("tests/basis/cc-pVDZ.gbs", id="cc-pVDZ"),
]

@pytest.mark.parametrize("molecule", TEST_MOLECULES)
@pytest.mark.parametrize("basis", TEST_BASIS)
def test_build_molecule(molecule, basis):
    build_molecule(molecule, basis)

@pytest.mark.parametrize("molecule", TEST_MOLECULES)
@pytest.mark.parametrize("basis", TEST_BASIS)
def test_integrals(molecule, basis):
    m, b = build_molecule(molecule, basis)
    mints =integrals(m,b)

    ovlp = mints.overlap()
    E_nuc =nuclear_repulsion(m)
    kinetic = mints.kinetic_energy()
    nuc_attr = mints.nuclear_attraction()
    # Takes too long
    #eri = mints.electron_repulsion()

