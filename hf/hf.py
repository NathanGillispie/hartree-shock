#!/usr/bin/env python
"""
Parser. Nothing special
"""
__author__ = "Nathan Gillispie"

import files_parser
import ints
import constants

def build_molecule(basis_file, mol_file):
    """Initialize within a module.
    Produces the basis and molecule objects.
        init(basis_file, molecule_file)
    Must be of formats Gaussian98 and XYZ.
    """
    long_basis = files_parser.parse_gbs(basis_file)
    mol = files_parser.parse_mol(mol_file)
    if (long_basis=={}):
        exit("Basis file empty...\n❌GAME OVER❌\nTotal score: 0 Hartrees")
    if (mol==[]):
        exit("MOL EMPTY... ugh (σ-σ)")

    basis = []
    molecule = []
    for atom in mol:
        atom_string = constants.Z_to_element[atom[0]-1]
        b_segment = long_basis[atom_string]
        num_primitives = 0
        for l,z,c in b_segment:
            for _ in range((l+1)*(l+2)//2):
                basis.append((l,z,c))
                num_primitives += 1
        molecule.append([atom[0], atom[1], num_primitives, b_segment])

    return molecule, basis

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="Hartree-Fock doer", description="Calculator of the Hartree-Focks")
    parser.add_argument('-b', '--basis', help='Filename of the basis set. Uses Gaussian98 format', default='basis/sto-3g.gbs')
    parser.add_argument('-m', '--mol', help='Filename of the molecule in XYZ format. See examples in tests.', default='tests/molecules/water.xyz')
    result = parser.parse_args()
    build_molecule(result.basis, result.mol)

