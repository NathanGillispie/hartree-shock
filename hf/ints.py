#!/usr/bin/env python
"""
File to compute nuclear repulsion, overlap, kinetic, and nuclear attraction integrals.
"""
import numpy as np
from math import erf
#dtype = numpy.complex128

Z = {}

def nuclear_repulsion(mol):
    natom = len(mol)
    energy = 0
    for i in range(natom):
        atom1_pos = mol[i][1]
        for j in range(i+1, natom):
            atom2_pos = mol[j][1]
            dist = np.sqrt(np.sum((atom2_pos-atom1_pos)**2))
            energy += mol[i][0]*mol[j][0]/dist
    return energy

def kinetic_energy(mol, basis):
    pass

def S_overlap(molecule, basis):
    nbf = len(basis)
    overlap_mat = np.zeros((nbf, nbf))

    coeffs, alphas, angs, basis_to_atom = [], [], [], []
    for b in basis:
        coeffs.append(b[2])
        alphas.append(b[1])
        angs.append(b[0])
    coeffs = np.array(coeffs)
    alphas = np.array(alphas)

    for i, a in enumerate(molecule):
        # a[0] is atom Z, a[2] is num_basis
        basis_to_atom += [i]*a[2]

    for i in range(nbf):
        a1 = alphas[i]
        c1 = coeffs[i]
        l1 = angs[i]
        atom1_pos = molecule[basis_to_atom[i]][1]
        for j in range(i, nbf):
            c2 = coeffs[j]
            a2 = alphas[j]
            l2 = angs[j]
            atom2_pos = molecule[basis_to_atom[j]][1]

            dist = np.sqrt(np.sum((atom2_pos-atom1_pos)**2))

            # assume len(coeffs) == len(alphas)
            for prim in range(c1.shape[0]):
                

            overlap_mat[i,j] = 0

def nuclear_attraction(mol, basis):
    pass

if __name__ == "__main__":
    import hf
    molecule, basis = hf.build_molecule('basis/sto-3g.gbs', 'tests/molecules/water.xyz')
    S_overlap(molecule, basis)
