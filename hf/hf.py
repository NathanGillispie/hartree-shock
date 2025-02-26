#!/usr/bin/env python
"""
/////////////////////
// HARTREE SHOCK ⚡//
/////////////////////

Simple Hartree-Fock program to compute my molecular orbitals for me.
"""
__author__ = "Nathan Gillispie"

import numpy as np

from utils import parse_mol, parse_gbs, molecular_grid, np2mathematica
import constants
from rhf import RHF

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

    save = True
    global wfn

    import pickle
    if save:
        wfn = RHF(molecule, basis)
        E, C = wfn.compute_E()
        with open("cinnom.pkl", "wb") as f:
            pickle.dump(wfn,f)
    else:
        with open("cinnom.pkl", "rb") as f:
            wfn = pickle.load(f)

    HOMO = wfn.occ
    AO2MO = wfn.C_MO.T

    # wfn.write_molden("h2o.molden")

    global grid
    grid = molecular_grid(wfn, spacing=.18, extension=3, transform=AO2MO)

    points, eval = grid.eval_basis_grid()
    X, Y, Z = tuple(points.T.reshape(np.insert(grid.shape, 0, 3)))
    
    values = np.log(eval[HOMO]**2)
    
    import plotly.graph_objects as go
    # fig= go.Figure(data=go.Isosurface(
    #     x=X.flatten(),
    #     y=Y.flatten(),
    #     z=Z.flatten(),
    #     value=values.flatten(),
    #     isomin=0.001,
    #     isomax=0.01,
    #     surface_count=2,
    #     colorbar_nticks=2
    # ))
    print("Plotting...")
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=-7,
        isomax=-2.5,
        opacity=0.12, # needs to be small to see through all surfaces
        surface_count=7
    ))
    fig.show()


