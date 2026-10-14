"""Step 13: all-length assembly with COMPILED small kernels.

DFT of length n:
  * n <= SMALL: the compiled fixed-pattern word compile_blocks(e^{2 pi i/n}, n)  (step 11c);
  * n = n1*n2 with gcd(n1,n2)=1: Good-Thomas/CRT index maps, sub-DFTs recursively;
  * otherwise (large prime power or prime): Bluestein chirp, whose inner power-of-two FFT is the
    lab radix-2 (NOT compiled; stated limitation).
Float64 reference. Checks the assembly for all n in a range against np.fft.
"""

import cmath
import math

import numpy as np

from allength import compiler as CP
from allength.dft_all import _coprime_split, bluestein

SMALL = 24
_cache = {}


def compiled_kernel(n):
    if n not in _cache:
        u = cmath.exp(-2j * cmath.pi / n)   # numpy convention: X_k = sum x_j e^{-2 pi i jk/n}
        _cache[n] = CP.compile_blocks(u, n)
    return _cache[n]


def assembled_dft(x):
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    if n == 1:
        return x.copy()
    if n <= SMALL:
        return CP.apply_blocks(compiled_kernel(n), x)
    split = _coprime_split(n)
    if split is not None:
        n1, n2 = split
        return _good_thomas(x, n1, n2)
    return bluestein(x)


def _good_thomas(x, n1, n2):
    n = n1 * n2
    u1 = pow(n2, -1, n1)
    u2 = pow(n1, -1, n2)
    i1 = np.arange(n1)[:, None]
    i2 = np.arange(n2)[None, :]
    A = x[(n2 * i1 + n1 * i2) % n]
    B = np.empty((n1, n2), dtype=np.complex128)
    for c in range(n2):
        B[:, c] = assembled_dft(A[:, c])
    for r in range(n1):
        B[r, :] = assembled_dft(B[r, :])
    k1 = np.arange(n1)[:, None]
    k2 = np.arange(n2)[None, :]
    out = (k1 * n2 * u1 + k2 * n1 * u2) % n
    X = np.empty(n, dtype=np.complex128)
    X[out] = B
    return X
