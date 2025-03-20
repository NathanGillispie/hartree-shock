#!/usr/bin/env python
"""\033[48:5:220m\033[38:5:0m
▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞
▞▞ HARTREE SHOCK ⚡▞▞
▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞
\033[0m
Simple Hartree-Fock program to compute my molecular orbitals for me.

Capabilities:
 - Restricted/unrestricted Hartree-Fock calculations
 - MO Isosurfaces via plotly
 - Molden file output (all MOs)
 - Grid file output (one MO)
 - libcint ints OR python ints via gbasis
 - Cool SCF progress bar, wow!
 - DIIS for faster convergence
"""
__author__ = "Nathan Gillispie"

import pickle
import numpy as np

from hs.utils import parse_mol, parse_gbs, molecular_grid, np2mathematica
from hs import constants
from hs.rhf import RHF
from hs.uhf import UHF

def build_molecule(mol_file, basis_file, **kwargs):
    r"""Initialize within a module. Returns tuple of molecule and basis. 
    Molecule is a list, each element is a tuple of (atomic_number, coord, basis_info)
    >>> build_molecule(Gaussian98_file, XYZ_file)
    (molecule, basis)
    """
    a_u = True
    if ("atomic_units" in kwargs.keys()):
        a_u = kwargs["atomic_units"]

    basis = parse_gbs(basis_file)
    mol = parse_mol(mol_file, atomic_units=a_u)

    if (basis=={}):
        exit("Basis file empty...\n❌GAME OVER❌\nTotal score: 0 Hartrees")
    if (mol==[]):
        exit("MOL EMPTY... (σ-σ) perhaps you used letters for the elements?")

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
    parser = argparse.ArgumentParser(prog="⚡Hartree-Shock⚡",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                description=__doc__)
    parser.add_argument('-b', '--basis',
                        help='Filename of the basis set. Uses Gaussian98 format',
                        default='sto-3g')
    parser.add_argument('-m', '--mol',
                        help='Filename of the molecule in XYZ format. See examples in tests.',
                        default='water')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--save',
                        help='Compute RHF and save pickle as filename.')
    group.add_argument('-l', '--load',
                        help='Load previous calculation from pickle.')
    return parser.parse_args()

def get_wfn(result):
    if result.load != None:
        try:
            with open(result.load, 'rb') as f:
                wfn = pickle.load(f)
            print("Loaded wavefunction from", result.load)
            return wfn
        except FileNotFoundError:
            print("Cannot load wavefunction from pickle")

    wfn_rhf = RHF(*build_molecule(result.mol, result.basis))
    E, C = wfn_rhf.compute_E()

    if result.save != None:
        try:
            with open(result.save, 'wb') as f:
                pickle.dump(wfn, f)
            print("Saved wavefunction as", result.save)
        except Exception as err:
            print(err)

    return wfn_rhf

def plot_density(wfn):
    try:
        import plotly.graph_objects as go
    except ImportError:
        exit("You must install plotly to plot the density")

    wfn.write_molden("h2o_rhf.molden")

    HOMO = wfn.occ-1 #-1 for indexing
    AO2MO = wfn.C_MO.T

    grid = molecular_grid(wfn, spacing=.15, extension=3, transform=AO2MO)

    points, eval = grid.eval_basis_grid()
    print("Evaluated basis grid for %d points"%points.shape[0])

    values = eval[HOMO+1]**2
    total_dens = np.sum(values)
    print("Total density = ", total_dens)
    values = np.log(values / total_dens) # normalize

    print("Plotting...")
    fig = go.Figure(data=go.Volume(
        x=grid.X.flatten(),
        y=grid.Y.flatten(),
        z=grid.Z.flatten(),
        value=values.flatten(),
        isomin=-7,
        isomax=-1,
        opacity=0.1,
        surface_count=12
    ))
    fig.show()

if __name__ == "__main__":
    wfn = get_wfn(parse())


