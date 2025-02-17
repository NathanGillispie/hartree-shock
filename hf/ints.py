#!/usr/bin/env python
"""
File to compute nuclear repulsion, overlap, kinetic, and nuclear attraction integrals.
Uses the gbasis library which uses libcint as the integral computer.
"""
import numpy as np
from gbasis.parsers import make_contractions, parse_gbs
from gbasis.integrals.libcint import ELEMENTS, LIBCINT, CBasis

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

class integrals:
    def __init__(self, molecule, basis):
        self.molecule = molecule
        self.basis = basis
        natoms = len(molecule)
        Z = [molecule[a][0] for a in range(natoms)]
        atnums = np.asarray(Z, dtype=float)
        atsyms = [ELEMENTS[atom] for atom in Z]
        atcoords = np.asarray([molecule[a][1] for a in range(natoms)])
        py_basis = make_contractions(basis, atsyms, atcoords, coord_types="cartesian")
        self.lc_basis = CBasis(py_basis, atsyms, atcoords, coord_type="cartesian")

    def overlap(self):
        return self.lc_basis.overlap_integral()
    def kinetic_energy(self):
        return self.lc_basis.kinetic_energy_integral()
    def nuclear_attraction(self):
        return self.lc_basis.nuclear_attraction_integral()
    def momentum(self):
        return self.lc_basis.momentum_integral(origin=np.zeros(3))
    def electron_repulsion(self):
        return self.lc_basis.electron_repulsion_integral()

