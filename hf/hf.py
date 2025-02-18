#!/usr/bin/env python
"""
/////////////////////
// HARTREE SHOCK ⚡//
/////////////////////

Simple Hartree-Fock program to compute my molecular orbitals for me.
"""
__author__ = "Nathan Gillispie"

import numpy as np
from gbasis.parsers import parse_gbs
from gbasis.evals.eval import evaluate_basis

from utils import parse_mol, grid_from_molecule
import ints
import constants
from rhf import RHF

def build_molecule(mol_file, basis_file):
    r"""Initialize within a module. Returns tuple of molecule and basis. 
    Molecule is a list, each element is a tuple of (atomic_number, coord, basis_info)
    >>> build_molecule(Gaussian98_file, XYZ_file)
    (molecule, basis)
    """
    basis = parse_gbs(basis_file)
    mol = parse_mol(mol_file)
    if (basis=={}):
        exit("Basis file empty...\n❌GAME OVER❌\nTotal score: 0 Hartrees")
    if (mol==[]):
        exit("MOL EMPTY... ugh (σ-σ)")

    molecule = []

    #strip the basis set to only contain the relevant atoms.
    relevant_basis = {}
    for atom in mol:
        atom_string = constants.Z_to_element[atom[0]]
        b_segment = basis[atom_string]
        if not atom_string in relevant_basis.keys():
            relevant_basis[atom_string] = b_segment

    return mol, relevant_basis

def parse():
    import argparse
    parser = argparse.ArgumentParser(prog="Hartree-Fock doer", 
                                     description="Calculator of the Hartree-Focks")
    parser.add_argument('-b', '--basis',
                        help='Filename of the basis set. Uses Gaussian98 format',
                        default='tests/basis/sto-3g.gbs')
    parser.add_argument('-m', '--mol',
                        help='Filename of the molecule in XYZ format. See examples in tests.',
                        default='tests/molecules/water.xyz')
    return parser.parse_args()


if __name__ == "__main__":
    result = parse()
    molecule, basis = build_molecule(result.mol, result.basis)

    wfn = RHF(molecule, basis)
    # E, C = wfn.compute_E()
    C = wfn.C
    HOMO = wfn.HOMO # in AO basis

    grid_from_molecule(molecule, basis)
    # TODO: get points from grid
    # evaluate_basis(basis, points, transform=C.T)



