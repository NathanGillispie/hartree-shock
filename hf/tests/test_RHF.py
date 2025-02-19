#!/usr/bin/env python
import pytest

from hf import build_molecule
from rhf import RHF

def test_water_E():
    basis = "tests/basis/sto-3g.gbs"
    molecule = "tests/molecules/water.xyz"
    m,b = build_molecule(molecule, basis)
    wfn = RHF(m,b)
    E, _ = wfn.compute_E()
    assert abs(E + 74.942079928192) < 2e-6

def test_methane_E():
    basis = "tests/basis/sto-3g.gbs"
    molecule = "tests/molecules/methane.xyz"
    m,b = build_molecule(molecule, basis)
    wfn = RHF(m,b)
    E, _ = wfn.compute_E()
    assert abs(E + 39.726850324347) < 2e-6
