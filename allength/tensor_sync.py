"""Step 15: synchronised tensor Fourier transform (synchronization.tex, proposition tensor-fourier).

Each axis j carries a compiled fixed-pattern block sequence for F_{r_j}. Blocks are padded with
identity blocks to a common length L (same pattern on every axis). Slot t of the tensor word is
the Kronecker product over axes of the axis-t block matrices. By the mixed-product property the
product over t equals the tensor product of the per-axis products, i.e. kron_j F_{r_j}.
"""

import cmath
import math

import numpy as np

from allength import compiler as CP


def axis_block_matrices(r):
    """Matrices of the blocks of compile_blocks(e^{-2 pi i/r}, r), each r x r."""
    u = cmath.exp(-2j * cmath.pi / r)
    blocks = CP.compile_blocks(u, r)
    mats = []
    for blk in blocks:
        M = np.zeros((r, r), dtype=np.complex128)
        for col in range(r):
            e = np.zeros(r, dtype=np.complex128)
            e[col] = 1
            M[:, col] = CP.apply_blocks([blk], e)
        mats.append(M)
    return mats


def identity_block(r):
    return np.eye(r, dtype=np.complex128)


def tensor_word(radices):
    """Kronecker-synchronised block sequence for kron_j F_{r_j}; returns list of full matrices."""
    per_axis = [axis_block_matrices(r) for r in radices]
    L = max(len(m) for m in per_axis)
    padded = [m + [identity_block(r)] * (L - len(m)) for m, r in zip(per_axis, radices)]
    words = []
    for t in range(L):
        W = np.array([[1.0 + 0j]])
        for j in range(len(radices)):
            W = np.kron(W, padded[j][t])
        words.append(W)
    return words, per_axis


def tensor_product_result(radices):
    words, _ = tensor_word(radices)
    total = np.eye(int(np.prod(radices)), dtype=np.complex128)
    for W in words:
        total = W @ total
    return total


def tensor_fourier(radices):
    mats = [np.array([[cmath.exp(-2j * cmath.pi * i * j / r) for j in range(r)] for i in range(r)])
            for r in radices]
    K = np.array([[1.0 + 0j]])
    for F in mats:
        K = np.kron(K, F)
    return K
