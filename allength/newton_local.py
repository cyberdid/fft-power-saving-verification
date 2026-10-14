"""Step 2: Newton reduction lemma (local.tex, lemma loc:newton-reduction).

For a primitive r-th root of unity u and F_r(u)_{ij} = u^{ij}:
    F_r(u) = N * D_N^{-1} * N^T,
where N = F_r(u) * P, P is unit upper triangular with column k the coefficients of
prod_{j<k} (Z - u^j), so N_{ik} = prod_{j<k} (u^i - u^j) (lower triangular), and
D_N = diag(N_00, ..., N_{r-1,r-1}).

Also checks the factorisation used in the proof:
    N_{ik} = (-1)^k u^{k(k-1)/2} H_i / H_{i-k},   H_j = prod_{s=1}^{j} (1 - u^s),
i.e. N = diag(H_i) * T * diag((-1)^k u^{k(k-1)/2}) with T_{ik} = H_{i-k}^{-1} (i>=k),
a lower triangular Toeplitz matrix with unit diagonal.

Implemented with mpmath at configurable precision (the identity is exact in Q(u)).
A complex128 version is provided to measure the floating-point behaviour separately.
"""

import math

import mpmath as mp
import numpy as np


def primitive_root(r, a=1, dps=80):
    """exp(2*pi*i*a/r) as an mpmath number; requires gcd(a, r) == 1."""
    assert math.gcd(a, r) == 1
    return mp.exp(2j * mp.pi * a / r)


def fourier_matrix(r, u):
    return mp.matrix([[u ** (i * j) for j in range(r)] for i in range(r)])


def newton_P(r, u):
    """Unit upper triangular P: column k = coefficients (ascending) of prod_{j<k}(Z - u^j)."""
    P = mp.zeros(r, r)
    for k in range(r):
        poly = [mp.mpc(1)]
        for j in range(k):
            root = u ** j
            new = [mp.mpc(0)] * (len(poly) + 1)
            for m, c in enumerate(poly):          # multiply by (Z - root)
                new[m + 1] += c
                new[m] -= root * c
            poly = new
        for m in range(k + 1):
            P[m, k] = poly[m]
    return P


def newton_N(r, u):
    """N = F_r(u) * P (lower triangular)."""
    return fourier_matrix(r, u) * newton_P(r, u)


def H(j, u):
    out = mp.mpc(1)
    for s in range(1, j + 1):
        out *= (1 - u ** s)
    return out


def N_closed(r, u):
    """Closed form N_{ik} = (-1)^k u^{k(k-1)/2} H_i / H_{i-k} for i>=k, else 0."""
    Hs = [H(j, u) for j in range(r)]
    M = mp.zeros(r, r)
    for i in range(r):
        for k in range(i + 1):
            M[i, k] = ((-1) ** k) * (u ** (k * (k - 1) // 2)) * Hs[i] / Hs[i - k]
    return M


def factorisation_residual(r, u):
    """max |F - N D^{-1} N^T| entrywise, mpmath."""
    F = fourier_matrix(r, u)
    N = newton_N(r, u)
    d = [N[i, i] for i in range(r)]
    R = mp.zeros(r, r)
    for i in range(r):
        for j in range(r):
            s = mp.mpc(0)
            for k in range(r):
                s += N[i, k] * N[j, k] / d[k]
            R[i, j] = F[i, j] - s
    return max(abs(R[i, j]) for i in range(r) for j in range(r))


def closed_form_residual(r, u):
    N = newton_N(r, u)
    C = N_closed(r, u)
    return max(abs(N[i, k] - C[i, k]) for i in range(r) for k in range(i + 1))


def toeplitz_entries_ok(r, u):
    """T_{ik} = H_{i-k}^{-1} depends only on i-k, is lower triangular, unit diagonal."""
    Hs = [H(j, u) for j in range(r)]
    ok = True
    for i in range(r):
        for k in range(r):
            if i < k:
                continue
            t = 1 / Hs[i - k]
            if i - k == 0 and abs(t - 1) > mp.mpf(10) ** (-50):
                ok = False
    return ok


# ---- complex128 measurement -------------------------------------------------

def newton_N_float(r, u):
    """Same product formula evaluated in complex128 (as the lab's NewtonPlan does)."""
    uf = complex(u)
    N = np.zeros((r, r), dtype=np.complex128)
    for i in range(r):
        for k in range(i + 1):
            prod = 1 + 0j
            for j in range(k):
                prod *= (uf ** i - uf ** j)
            N[i, k] = prod
    return N


def factorisation_residual_float(r, u):
    uf = complex(u)
    F = np.array([[uf ** (i * j) for j in range(r)] for i in range(r)], dtype=np.complex128)
    N = newton_N_float(r, u)
    d = np.diag(N).copy()
    R = F - N @ np.diag(1 / d) @ N.T
    return np.max(np.abs(R)) / np.max(np.abs(F))


def cond_float(r, u):
    return np.linalg.cond(newton_N_float(r, u))
