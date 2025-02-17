#!/usr/bin/env python
"""
Parser. Nothing special
"""
__author__ = "Nathan Gillispie"

import numpy as np
from time import perf_counter

import files_parser
import ints
import constants

def build_molecule(basis_file, mol_file):
    r"""Initialize within a module. Returns tuple of molecule and basis. 
    Molecule is a list, each element is a tuple of (atomic_number, coord, basis_info)
    >>> build_molecule(Gaussian98_file, XYZ_file)
    (molecule, basis)
    """
    long_basis = files_parser.parse_gbs(basis_file)
    mol = files_parser.parse_mol(mol_file)
    if (long_basis=={}):
        exit("Basis file empty...\n❌GAME OVER❌\nTotal score: 0 Hartrees")
    if (mol==[]):
        exit("MOL EMPTY... ugh (σ-σ)")

    # This basis is separated by atom then subshell, not used currently
    basis = []
    molecule = []
    for atom in mol:
        atom_string = constants.Z_to_element[atom[0]]
        b_segment = long_basis[atom_string]
        num_primitives = 0
        for l,z,c in b_segment:
            for _ in range((l+1)*(l+2)//2):
                basis.append((l,z,c))
                num_primitives += 1
        molecule.append([atom[0], atom[1], num_primitives, b_segment])

    return molecule, long_basis

def parse():
    import argparse
    parser = argparse.ArgumentParser(prog="Hartree-Fock doer", description="Calculator of the Hartree-Focks")
    parser.add_argument('-b', '--basis', help='Filename of the basis set. Uses Gaussian98 format', default='basis/sto-3g.gbs')
    parser.add_argument('-m', '--mol', help='Filename of the molecule in XYZ format. See examples in tests.', default='tests/molecules/water.xyz')
    return parser.parse_args()

if __name__ == "__main__":
    result = parse()
    molecule, basis = build_molecule(result.basis, result.mol)

    # Compute integrals
    mints = ints.integrals(molecule, basis)

    start = perf_counter()
    e_nuc = ints.nuclear_repulsion(molecule)
    ovlp = mints.overlap()
    kinetic = mints.kinetic_energy()
    nuc_attr = mints.nuclear_attraction()
    eri = mints.electron_repulsion()
    print("Integrals computed: %.3fms"%(1000*(perf_counter() - start)))

    start = perf_counter()
    Lambda_S, L_S = np.linalg.eigh(ovlp)
    S_inv = L_S @ np.diag(1/np.sqrt(Lambda_S)) @ np.transpose(L_S)
    print("Diagonalized overlap: %.3fms"%(1000*(perf_counter() - start)))

    print(np.array2string(S_inv, precision=6, suppress_small=True, max_line_width=200))
