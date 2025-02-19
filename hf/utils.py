#!/usr/bin/env python

__credits__ = ["https://github.com/theochem/gbasis"]
__license__ = "GPL"
__author__ = "Taewon David Kim, Leila Pujal, et. al."

import numpy as np

def parse_mol(filename, atomic_units=True):
    """Parse a file into coords as a dictionary using XYZ format.
    parse_mol(filename)
    Comments are not included.
    Written by Nathan Gillispie
    """

    Z = []
    coords = []
    with open(filename, 'r') as mol_file:
        a = mol_file.read()
        a = a.split('\n')
        # skip the first line
        for line in a[1:]:
            s = line.split(' ')
            s = [f for f in s if f!='']
            if (len(s) == 4):
                Z.append(int(s[0]))
                coords.append([float(s[1]), float(s[2]), float(s[3])])
    coords = np.asarray(coords)

    # CENTER OF MASS CALCULATION
    # It's misleading but the ratio is close enough
    masses = np.asarray(Z,dtype=float)
    com = (masses @ coords)/masses.sum()
    for i in range(len(coords)):
        coords[i] -= com
        if not atomic_units:
            coords[i] /= .5291772105
    return list(zip(Z, coords))

def write_mos(C, nel:int, filename:str) -> None:
    """Save molecular orbitals to file.
    >>> write_mos(C_MO, num_electrons, filename)
    """
    C = np.asarray(C)
    nbf = len(C[0]) 
    nmo = len(C[0][0]) 
    header = bytearray(np.array([nbf,nmo,nel[0],nel[1]]))
    with open(filename,'wb') as f:
        f.write(header)
        C = C.reshape(2*nbf*nmo)
        for val in C:
            f.write(bytearray(val))
    print("File written to %s!"%mos_filename)

def grid_from_molecule(molecule, ao_basis, transform=None, spacing=0.2, extension=4.0):
    """Produces the grid given a molecule, ao_basis, and transform"""
    # Rotate the molecule
    Z, coords = zip(*molecule)
    masses = np.asarray(Z, dtype=float)

    inertia = np.zeros([3,3])
    for i in range(masses.shape[0]):
        pos = coords[i]
        r = pos[0]**2 + pos[1]**2 + pos[2]**2
        inertia += masses[i] * (np.diag([r,r,r]) - np.outer(pos.T, pos))
    _, vecs = np.linalg.eigh(inertia)
    new_coords = np.dot(coords, vecs)
    axes = spacing * vecs

    max = np.amax(new_coords, axis=0)
    min = np.amin(new_coords, axis=0)

    shape = np.array(np.ceil((max - min + 2.0 * extension) / spacing), int)
    origin = np.dot((-0.5 * shape), axes)
    dim = 3 #origin.shape
    # origin axes shape weight="trapezoid"

    points = np.vstack(np.meshgrid(
                shape[0],
                shape[1],
                shape[2],
                indexing="ij")).reshape(3, -1).T
    return points

def write_cube(filename, data, atcoords):
    with open(filename, 'w') as f:
        f.write("Cubefile created with THEOCHEM Grid\n")
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        x, y, z = self._origin
        f.write(f"{natom:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
        rvecs = self._axes
        for i, (x, y, z) in zip(self._shape, rvecs):
            f.write(f"{i:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
        for i, q, (x, y, z) in zip(atnums, pseudo_numbers, atcoords):
            f.write(f"{i:5d} {q:11.6f} {x:11.6f} {y:11.6f} {z:11.6f}\n")
        # writing the cube data:
        num_chunks = 6
        for i in range(0, data.size, num_chunks):
            row_data = data.flat[i : i + num_chunks]
            f.write((row_data.size * " {:12.5E}").format(*row_data))
            f.write("\n")

