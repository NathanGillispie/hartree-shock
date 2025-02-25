import pytest
from gbasis.parsers import make_contractions
import numpy as np
from gbasis.integrals.electron_repulsion import ElectronRepulsionIntegral

import ints
from utils import parse_gbs, parse_mol
import wavefunction

@pytest.fixture(autouse=True)
def ints_inorg():
    """Random inorganic molecule"""
    basis = parse_gbs("""
! required space
H     0
S    1   1.00
      0.1612777588D+00       1.0000000
****
Ni     0
SP   1   1.00
      0.4519500000D-01       0.1000000000D+01       0.1000000000D+01
D    2   1.00
      0.2819147000D+02       0.9098880504D-01
****
""")
    molecule = parse_mol("tests/molecules/NiH2.xyz")
    
    gbasis_ints  = wavefunction.wavefunction(molecule, basis, use_libcint=False).ints
    libcint_ints = wavefunction.wavefunction(molecule, basis, use_libcint=True).ints
    
    return gbasis_ints, libcint_ints

def test_overlap(ints_inorg):
    gbas, libc = ints_inorg
    assert np.allclose(gbas["overlap"], libc["overlap"])

def test_nuclear(ints_inorg):
    gbas, libc = ints_inorg
    assert np.allclose(gbas["nuclear"], libc["nuclear"])

def test_kinetic(ints_inorg):
    gbas, libc = ints_inorg
    assert np.allclose(gbas["kinetic"], libc["kinetic"])

def test_eri(ints_inorg):
    gbas, libc = ints_inorg
    assert np.allclose(gbas["eri"], libc["eri"])


def test_F_ERI():
    """Test F overlap for ERI"""
    basis = parse_gbs("""
! Required space
Na     0
F    1   1.00
      0.1614750979D+00       0.9003984260D+00
****
    """)
    molecule = [(11, np.array([0.,0.,0.])), (11, np.array([1.,0.,0.]))]

    shells = make_contractions(basis, ["Na", "Na"], np.array([[0.,0.,0.], [1.,0.,0.]]), "cartesian")
    gbasis_eri = ElectronRepulsionIntegral(shells).construct_array_cartesian()

    mints = ints.integrals(molecule, basis, True)
    libcint_eri = mints.electron_repulsion()

    assert np.allclose(gbasis_eri, libcint_eri)
