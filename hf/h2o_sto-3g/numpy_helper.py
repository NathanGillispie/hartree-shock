#!/usr/bin/env python

"""
Puts mats into numpy format
"""

import numpy as np

__all__ = ["mat_to_np"]

def mat_to_np(mat_file, shape=(-1,-1)):
    mat = None
    with open(mat_file, 'r') as f:
        mat = np.loadtxt(f)
        mat = np.transpose(mat)
    
    rows = np.array(mat[0], dtype=int)
    cols = np.array(mat[1], dtype=int)
    dim = max(rows.max(), cols.max())

    out = np.zeros((dim, dim))
    for k in range(mat.shape[1]):
        i = cols[k]
        j = rows[k]
        out[i-1][j-1] = mat[2][k]

    return out

def print_mat(mat):
    print(np.array2string(arr, precision=6, suppress_small=True, max_line_width=200))

if __name__ == "__main__":
    arr = mat_to_np("s.dat")
    print_mat(arr)
