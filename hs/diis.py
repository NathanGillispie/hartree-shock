
import numpy as np

class DIIS():
    """
    Direct Inversion in the Iterative Subspace (DIIS)

    A new fock matrix is extrapolated from the past few fock matricies
    during the SCF procedure.
    """
    def __init__(self, wfn, keep=8):
        """
        DIIS initialization takes a wavefunction object.
        The option keep determines the number of error 
        matricies that are kept to derive the new fock matrix.
        The default is 8.
        """
        self.ovlp = wfn.ints["overlap"]
        self.debug = wfn.debug
        self.keep = keep

        # Can't declare the matrix without knowing the shape.
        self.F = []
        self.e = []

    def compute_F(self, F, D):
        if len(F.shape)!=2:
            exit("DIIS: WRONG F SHAPE")
        if len(D.shape)!=2:
            exit("DIIS: WRONG D SHAPE")

        # Compute error matrix
        self.e.append(F @ D @ self.ovlp - self.ovlp @ D @ F)
        self.F.append(F)

        if len(self.e) > self.keep:
            # First in, first out
            self.e = self.e[1:]
            self.F = self.F[1:]

        # Compute B_ij as the dot product of matricies e_i and e_j
        err = np.asarray(self.e)
        B = np.einsum("iab,jab->ij",err, err, optimize=True)
        B = np.pad(B, (0,1),constant_values=-1)
        B[-1,-1] = 0
        # Set up linear system
        sln = np.zeros((B.shape[0]))
        sln[-1] = -1
        # Solve B ci = sln
        ci = np.linalg.solve(B, sln)

        # Last element is the lagrange multiplier, remove it
        ci = ci[:-1]

        # New F is a linear combination of previous F
        # F = \sum_i c_i F_i
        return np.einsum("bij,b",np.asarray(self.F),ci,optimize=True)

