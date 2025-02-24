import pytest
from gbasis.parsers import make_contractions
import numpy as np
from gbasis.integrals.electron_repulsion import ElectronRepulsionIntegral

import constants
import ints
from utils import parse_gbs

def test_libcint_SPD():
    """Random basis with S, P and D overlap"""
    basis = parse_gbs("""
! Required space
Na     0
S    1   1.00
      0.9105710923D+00       0.1114824971D+00
P    1   1.00
      0.1236238776D+02       0.4446345422D+00
D    1   1.00
      0.1614750979D+00       0.9003984260D+00
****
    """)
    molecule = [(11, np.array([0.,0.,0.]))]

    shells = make_contractions(basis, ["Na"], np.array([[0.,0.,0.]]), "cartesian")
    gbasis_eri = ElectronRepulsionIntegral(shells).construct_array_cartesian()

    mints = ints.integrals(molecule, basis)
    libcint_eri = mints.electron_repulsion()

    assert np.allclose(gbasis_eri, libcint_eri)

def test_libcint_F():
    """Random basis with F overlap"""
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

    mints = ints.integrals(molecule, basis)
    libcint_eri = mints.electron_repulsion()

    assert np.allclose(gbasis_eri, libcint_eri)
