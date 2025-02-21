#!/usr/bin/env python

__credits__ = ["https://github.com/theochem/gbasis"]
__license__ = "GPL"
__author__ = "Taewon David Kim, Leila Pujal, et. al."

import numpy as np
from gbasis.evals.eval import evaluate_basis
from gbasis.parsers import make_contractions
from gbasis.integrals.libcint import ELEMENTS

def parse_mol(f, atomic_units=True):
    """Parse a file into coords as a dictionary using XYZ format.
    parse_mol(f, atomic_units=True)

    Numbers are delimited by whitespace.
    """
    try:
        mol_file = open(f, 'r')
    except FileNotFoundError:
        print("File not found, trying to parse molecule directly from input.")
        import io
        mol_file = io.StringIO(f)

    Z = []
    coords = []

    a = mol_file.read()
    a = a.split('\n')

    for l_num, line in enumerate(a):
        s = line.split()
        s = [f for f in s if f!='']
        if (len(s) >= 4):
            try:
                Z.append(round(float(s[0])))
                coords.append([float(s[1]), float(s[2]), float(s[3])])
            except ValueError:
                print("Error parsing molecule: \"%s\" at line %d =\n    \"%s\""%(mol_file.name,l_num," ".join(s)))
    coords = np.asarray(coords)
    mol_file.close()

    # CENTER OF MASS CALCULATION
    # It's misleading but the ratio is close enough
    masses = np.asarray(Z,dtype=float)
    com = (masses @ coords)/masses.sum()
    for i in range(len(coords)):
        coords[i] -= com
        if not atomic_units:
            coords[i] /= .5291772105
    return list(zip(Z, coords))

def write_mos(C, nel, filename:str) -> None:
    """Save molecular orbitals to file.
    >>> write_mos(C_MO, num_electrons, filename)
    num_electrons: array-like (alpha, beta)
    """
    C = np.asarray(C).reshape(2*nbf*nmo)
    nbf = len(C[0])
    nmo = len(C[0][0])
    header = np.array([nbf,nmo,nel[0],nel[1]])
    with open(filename,'wb') as f:
        f.write(bytearray(header))
        for val in C:
            f.write(bytearray(val))
    print("File written to %s!"%mos_filename)

class molecular_grid:
    def __init__(self, wfn, transform=None, spacing=0.2, extension=4.0):
        self.wfn = wfn
        self.transform = transform
        if (transform == None):
            self.transform = wfn.S_inv
        self.spacing = spacing
        self.extension = extension

        self.grid_points, self.eval_points = self.eval_basis_grid()

    def eval_basis_grid(self):
        """Produces the grid given a molecule, ao_basis, and transform"""
        # Rotate the molecule, makes graphs look pretty
        Z, coords = zip(*self.wfn.molecule)
        masses = np.asarray(Z, dtype=float)

        inertia = np.zeros([3,3])
        for i in range(masses.shape[0]):
            pos = coords[i]
            r = pos[0]**2 + pos[1]**2 + pos[2]**2
            inertia += masses[i] * (np.diag([r,r,r]) - np.outer(pos.T, pos))
        _, vecs = np.linalg.eigh(inertia)
        new_coords = np.dot(coords, vecs)
        axes = self.spacing * vecs
        self._axes = axes

        max = np.amax(new_coords, axis=0)
        min = np.amin(new_coords, axis=0)

        self.shape = np.array(np.ceil((max - min + 2.0 * self.extension) / self.spacing), int)
        self._origin = np.dot((-0.5 * self.shape), axes)

        min = self._origin
        max = self._origin + self.spacing*self.shape

        self.X = np.linspace(min[0], max[0], num=self.shape[0], endpoint=False)
        self.Y = np.linspace(min[1], max[1], num=self.shape[1], endpoint=False)
        self.Z = np.linspace(min[2], max[2], num=self.shape[2], endpoint=False)

        grid_points = np.vstack(np.meshgrid(self.X,self.Y,self.Z,indexing="ij")).reshape(3, -1).T

        atoms = [ELEMENTS[a] for a in Z]
        contractions = make_contractions(self.wfn.basis, atoms, new_coords, coord_types="cartesian")
        eval_points = evaluate_basis(contractions, grid_points, transform=self.transform)

        # grid_points = grid_points.reshape((*self.shape,3))
        # nbf = eval_points.shape[0]
        # eval_points = eval_points.reshape((nbf,*self.shape))

        return grid_points, eval_points

    def grid_eval_x(self, occ):
        r"""Return the grid and points closest to the YZ-plane"""
        nbf = self.eval_points.shape[0]
        eval = self.eval_points.reshape((nbf,*self.shape))
        eval = eval[occ,eval.shape[1]//2,:,:]
        return self.Y, self.Z, eval.T

    def grid_eval_y(self, occ):
        r"""Return the grid and points closest to the XZ-plane"""
        nbf = self.eval_points.shape[0]
        eval = self.eval_points.reshape((nbf,*self.shape))
        eval = eval[occ,:,eval.shape[2]//2,:]
        return self.X, self.Z, eval.T

    def grid_eval_z(self, occ):
        r"""Return the grid and points closest to the XY-plane"""
        nbf = self.eval_points.shape[0]
        eval = self.eval_points.reshape((nbf,*self.shape))
        eval = eval[occ,:,:,eval.shape[3]//2]
        return self.X, self.Y, eval.T

    def write_cube(self, filename):
        data = self.eval_points
        atnums, atcoords = zip(*self.wfn.molecule)
        natom = len(atnums)
        pseudo_numbers = list(atnums)

        with open(filename, 'w') as f:
            f.write("Cubefile created with THEOCHEM Grid\n")
            f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
            x, y, z = self._origin
            f.write(f"{natom:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
            rvecs = self._axes
            for i, (x, y, z) in zip(self.shape, rvecs):
                f.write(f"{i:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
            for i, q, (x, y, z) in zip(atnums, pseudo_numbers, atcoords):
                f.write(f"{i:5d} {q:11.6f} {x:11.6f} {y:11.6f} {z:11.6f}\n")
            # writing the cube data:
            num_chunks = 6
            for i in range(0, data.size, num_chunks):
                row_data = data.flat[i : i + num_chunks]
                f.write((row_data.size * " {:12.5E}").format(*row_data))
                f.write("\n")

    def get_grid_eval(self):
        return self.grid_points, self.eval_points

def np2mathematica(arr):
    """Places numpy array in Windows clipboard in mathematica format"""
    from os import system
    s = np.array2string(arr, precision=7, suppress_small=True, separator=",", threshold=1000000)
    s = s.replace('[', '{').replace(']', '}')
    # Use WSL 🤖
    system(f"echo -n \"{s}\" | /mnt/c/Windows/System32/clip.exe")

class SCFConvergeError(Exception):
    """When SCF does not converge. Returns last two MOs"""
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg

