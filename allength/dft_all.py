"""Step 1: DFT of arbitrary length n from coprime factorisation (Good-Thomas/CRT)
and chirp convolution (Bluestein).

Scope: float64 / complex128 reference implementation of the algebraic blocks only.
Not the paper's fast uniform compiler, not the synthesis printer, not the
power-saving construction. Those are later steps.

Convention: X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n) (unnormalised, matches np.fft.fft).
"""

import math
import os
import sys

import numpy as np

# Reuse the lab's own radix-2 for power-of-two sizes inside Bluestein.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from numerical import radix2_fft  # noqa: E402

DIRECT_LIMIT = 64


def small_dft(x):
    """Direct DFT matrix; used only for short lengths (local kernel)."""
    n = len(x)
    j = np.arange(n)
    W = np.exp(-2j * np.pi * np.outer(j, j) / n)
    return W @ x


def _coprime_split(n):
    """Return (a, b) with a*b == n, gcd(a, b) == 1, a > 1, b > 1, or None."""
    # Split off one full prime power.
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            q = 1
            while m % p == 0:
                m //= p
                q *= p
            if m > 1:
                return q, n // q
            return None
        p += 1
    return None


def good_thomas(x, n1, n2):
    """DFT of length n = n1*n2 with gcd(n1, n2) == 1 via CRT index maps, no twiddles."""
    n = n1 * n2
    assert len(x) == n and math.gcd(n1, n2) == 1
    u1 = pow(n2, -1, n1) if n1 > 1 else 0  # inverse of n2 mod n1
    u2 = pow(n1, -1, n2) if n2 > 1 else 0  # inverse of n1 mod n2

    i1 = np.arange(n1)[:, None]
    i2 = np.arange(n2)[None, :]
    in_idx = (n2 * i1 + n1 * i2) % n            # CRT input map
    A = x[in_idx]                                # shape (n1, n2)

    # 2-D DFT along each axis with the length-n1 and length-n2 DFTs.
    B = np.empty((n1, n2), dtype=np.complex128)
    for c in range(n2):
        B[:, c] = dft_all(A[:, c])
    for r in range(n1):
        B[r, :] = dft_all(B[r, :])

    k1 = np.arange(n1)[:, None]
    k2 = np.arange(n2)[None, :]
    out_idx = (k1 * n2 * u1 + k2 * n1 * u2) % n  # CRT output map
    X = np.empty(n, dtype=np.complex128)
    X[out_idx] = B
    return X


def bluestein(x):
    """Chirp-z (Bluestein) DFT for any length using the lab's radix-2 FFT."""
    n = len(x)
    if n == 1:
        return x.astype(np.complex128)
    j = np.arange(n, dtype=np.int64)
    # Exact phase reduction: j^2 mod 2n keeps the argument small.
    c = np.exp(-1j * np.pi * ((j * j) % (2 * n)) / n)     # exp(-i*pi*j^2/n)
    d = np.exp(1j * np.pi * ((np.arange(-(n - 1), n) ** 2) % (2 * n)) / n)
    M = 1
    while M < 2 * n - 1:
        M <<= 1
    a = np.zeros(M, dtype=np.complex128)
    a[:n] = x * c
    b = np.zeros(M, dtype=np.complex128)
    # b[m] = d_{m} for m in [0, n) and b[M - m] = d_{-m} for m in [1, n)
    b[:n] = d[n - 1:]
    b[M - (n - 1):] = d[:n - 1]
    conv = _ifft_via_radix2(radix2_fft(a) * radix2_fft(b))
    return c * conv[:n]


def _ifft_via_radix2(y):
    """Inverse of length M (power of two) using only the forward radix-2."""
    M = len(y)
    return np.conj(radix2_fft(np.conj(y))) / M


def dft_all(x, axis=None):
    """DFT of arbitrary length; the 1-D input path is the one tested."""
    x = np.asarray(x, dtype=np.complex128)
    if axis is not None and x.ndim > 1:
        return np.apply_along_axis(dft_all, axis, x)
    n = len(x)
    if n <= DIRECT_LIMIT:
        return small_dft(x)
    split = _coprime_split(n)
    if split is not None:
        n1, n2 = split
        return good_thomas(x, n1, n2)
    return bluestein(x)
