#!/usr/bin/env python
"""
Parser. Nothing special
"""
__author__ = "Nathan Gillispie"

import numpy as np
from scipy.linalg import eigh
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

def occupied_orbitals(molecule, charge, multiplicity):
    Z_total = np.asarray([atom[0] for atom in molecule],dtype=int).sum()
    num_elec = Z_total - charge
    unpaired_elec = multiplicity - 1
    # n_elec + unpaired pairs all electrons to form whole MOs
    if ((num_elec + unpaired_elec) % 2 != 0):
        exit("Incompatible charge and multiplicity")
    occ = (num_elec + unpaired_elec)//2
    nbf = np.asarray([atom[2] for atom in molecule],dtype=int).sum()

    if (occ > nbf):
        exit("More occupied MOs than basis functions. This should never happen. Good luck.")
    return occ

if __name__ == "__main__":
    result = parse()
    molecule, basis = build_molecule(result.basis, result.mol)

    occupied_MO = occupied_orbitals(molecule, charge:=0, multiplicity:=1)

    # Compute integrals
    mints = ints.integrals(molecule, basis)

    start = perf_counter()
    ovlp = mints.overlap()
    E_nuc = ints.nuclear_repulsion(molecule)
    kinetic = mints.kinetic_energy()
    nuc_attr = mints.nuclear_attraction()
    eri = mints.electron_repulsion()
    H_core = nuc_attr + kinetic
    print("Integrals computed: %.3fms"%(1000*(perf_counter() - start)))

    start = perf_counter()
    Lambda_S, L_S = np.linalg.eigh(ovlp)
    S_inv = L_S @ np.diag(1/np.sqrt(Lambda_S)) @ np.transpose(L_S)
    print("Diagonalized overlap: %.3fms"%(1000*(perf_counter() - start)))

    start = perf_counter()
    F_0 = np.transpose(S_inv) @ H_core @ S_inv
    # ASSUMING F_0 is hermitian
    #epsilon_0, C_0 = np.linalg.eigh(F_0)
    epsilon_0, C_0 = eigh(F_0)
    C_0 = S_inv @ C_0 # transform to the AO basis
    print("Initial orbital energies: %sEh"%np.array2string(epsilon_0,precision=3,max_line_width=150,suppress_small=True,separator=''))
    print("Formed initial Fock: %.3fms"%(1000*(perf_counter() - start)))

    start = perf_counter()
    D_0 = np.einsum("mk,nk->mn",C_0[:,:occupied_MO],C_0[:,:occupied_MO],optimize=True)
    print("Formed initial density: %.3fms"%(1000*(perf_counter() - start)))

    E_elec = np.einsum("mn,mn",D_0,(H_core + F_0))
    E_tot = E_elec + E_nuc
    print("\nInitial energy: %.6fEh"%E_tot)

    #Starting SCF
    D = D_0
    occ = occupied_MO
    delta_E = 1
    previous_E = E_tot
    iteration = 0
    while(delta_E > 1e-6):
        start = perf_counter()
        iteration += 1

        # Compute new fock by minimizing electronic repulsion
        tmp = 2*eri - eri.transpose((0,2,1,3))
        # Using PREVIOUS density
        F_MO = np.einsum("ls, mnls->mn",D, tmp, optimize=True) + H_core
        F = S_inv.T @ F_MO @ S_inv          # Orthogonalize fock
        epsilon, C_MO = eigh(F)   # Diagonalize fock
        C = S_inv @ C_MO                    # Back transform

        # Compute new density
        D = np.einsum("mk,nk->mn",C[:,:occ],C[:,:occ],optimize=True)
        E_elec = (D @ (H_core + F)).sum()

        delta_E = abs(E_elec + E_nuc - E_tot)
        E_tot = E_elec + E_nuc # Previous iteration's Etot above

        # Show that the fock matrix is diagonal in MO basis
        F_ij = np.einsum("mj,ni,mn->ij",C,C,F_MO, optimize=True)
        F_ij = np.abs(F_ij-F_ij.T)
        if (F_ij.sum() > 1e-7):
            print("Fock matrix is not orthogonal in the MO basis!")

        print("Iter%3d: %.6f Eh  %.3fms"%(iteration, E_tot, 1000*(perf_counter()-start)))
        if (iteration > 200):
            exit("\nSCF did not converge")

    #print(np.array2string(C_0, precision=6, suppress_small=True, max_line_width=200))
