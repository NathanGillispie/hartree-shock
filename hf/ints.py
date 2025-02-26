#!/usr/bin/env python
"""
File to compute nuclear repulsion, overlap, kinetic, and nuclear attraction
integrals. Uses the gbasis library which uses libcint as the integral computer.
Optionally, you can disable libcint and use the gbasis integral computer in
pure python but this is much slower.
"""
import numpy as np
from gbasis.parsers import make_contractions, parse_gbs
from gbasis.integrals.libcint import ELEMENTS, CBasis

from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.kinetic_energy import KineticEnergyIntegral
import gbasis.integrals.nuclear_electron_attraction as nuc_attr
from gbasis.integrals.electron_repulsion import ElectronRepulsionIntegral

# ('only spinner', dict(bar=None, monitor=False, elapsed=False, stats=False))

def nuclear_repulsion(molecule):
    natom = len(molecule)
    energy = 0
    Z, coords = zip(*molecule)
    for i in range(natom):
        atom1_pos = coords[i]
        for j in range(i+1, natom):
            atom2_pos = coords[j]
            dist = np.sqrt(np.sum((atom2_pos-atom1_pos)**2))
            energy += Z[i]*Z[j]/dist
    return energy

class integrals:
    def __init__(self, molecule, basis, use_libcint):
        self.molecule = molecule
        self.basis = basis
        self.use_libcint = use_libcint

        natoms = len(molecule)
        Z, coords = zip(*molecule)
        atcoords = np.asarray(coords, dtype=float)
        atnums = np.asarray(Z, dtype=float)
        atsyms = [ELEMENTS[atom] for atom in Z]

        self.shells = make_contractions(basis, atsyms, atcoords, coord_types="cartesian")

        self.lc_basis = None
        self.lc_basis = CBasis(self.shells, atsyms, atcoords, coord_type="cartesian")
        if use_libcint:
            self.lc_basis = CBasis(self.shells, atsyms, atcoords, coord_type="cartesian")
        else:
            self.nuclear_coords = atcoords
            self.nuclear_charges = np.asarray(Z)

    def overlap(self):
        if self.use_libcint:
            return self.lc_basis.overlap_integral()
        else:
            return overlap_integral(self.shells)
    def kinetic_energy(self):
        if self.use_libcint:
            return self.lc_basis.kinetic_energy_integral()
        else:
            kinetic = KineticEnergyIntegral(self.shells)
            return kinetic.construct_array_cartesian()
    def nuclear_attraction(self):
        if self.use_libcint:
            return self.lc_basis.nuclear_attraction_integral()
        else:
            return nuc_attr.nuclear_electron_attraction_integral(
                    self.shells, self.nuclear_coords, self.nuclear_charges)
    def electron_repulsion(self):
        if self.use_libcint:
            return self.lc_basis.electron_repulsion_integral(notation="chemist")
        else:
            eri = ElectronRepulsionIntegral(self.shells)
            return eri.construct_array_cartesian()


    def nbf(self):
        if self.use_libcint:
            return self.lc_basis.nbfn
        else:
            n_funcs = [s.num_cart for s in self.shells]
            return sum(n_funcs)
    def nshells(self):
        return len(self.shells)


