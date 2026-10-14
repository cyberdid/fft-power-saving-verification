"""Step 5: lower triangular Toeplitz apply via block splitting, displacement reconstruction
of cross blocks, and convolution-style outer-product application (local.tex, loc:toeplitz).

Scope of this step (algebra only, float64/complex128 reference):
  * reciprocal series g (inverse of a lower triangular Toeplitz matrix)     [loc:reciprocal]
  * block split T_v = [[I,0],[E,I]] diag(T_p, T_{v-p}),  E = B T_p^{-1}       [loc:cross]
  * displacement entries E_ij - E_{i-1,j-1} = -h_{i-p} g_{p-j}               [loc:displacement-entry]
  * chunk reconstruction D = v1 w1^T + e0 R0^T + u e0^T, M = sum_l J^l D (J^T)^l [loc:displacement, loc:reconstruction]
  * application of an outer product by two convolutions                       [loc:two-convolutions]
  * recursive apply T x                                                       [block decomposition]

NOT in this step: the exact-width placement of gate coordinates inside the parent block (the
workspace rule that the lemma needs), and the round count O(1 + log^4 r). Those remain open.
"""

import numpy as np


def toeplitz_matrix(h, v):
    """Lower triangular Toeplitz T_v with T[i, j] = h[i-j] for i >= j."""
    T = np.zeros((v, v), dtype=np.complex128)
    for i in range(v):
        for j in range(i + 1):
            T[i, j] = h[i - j]
    return T


def reciprocal_series(h, r):
    """g_0 = 1/h_0, g_k = -(1/h_0) sum_{j=1}^k h_j g_{k-j}."""
    g = np.zeros(r, dtype=np.complex128)
    g[0] = 1 / h[0]
    for k in range(1, r):
        g[k] = -(1 / h[0]) * sum(h[j] * g[k - j] for j in range(1, k + 1))
    return g


def cross_entry(h, g, p, i, j):
    """E_ij = sum_{nu=j}^{p-1} h_{i-nu} g_{nu-j},  p <= i, 0 <= j < p."""
    return sum(h[i - nu] * g[nu - j] for nu in range(j, p))


def outer_apply(v, w, x, a):
    """Apply sum_l J^l (v w^T) (J^T)^l to x, returning a vector of length a.
    b_l = sum_{j>=l} w_{j-l} x_j ;  c_i = sum_{l} v_{i-l} b_l (zero extension)."""
    e = len(x)
    b = np.zeros(e, dtype=np.complex128)
    for l in range(e):
        b[l] = sum(w[j - l] * x[j] for j in range(l, e))
    c = np.zeros(a, dtype=np.complex128)
    for i in range(a):
        c[i] = sum(v[i - l] * b[l] for l in range(e) if 0 <= i - l < a)
    return c


def chunk_displacement(h, g, p, i0, a, j0, e):
    """Return the three outer products (v_k, w_k) with
    D = v1 w1^T + e0 R0^T + u e0^T for the chunk E[i0:i0+a, j0:j0+e], using only the first
    row and first column of the chunk (computed from the formula) and the interior rank-one part."""
    # interior rank-one part: D_rc = -h_{i0+r-p} g_{p-j0-c}
    v1 = np.array([-h[i0 + r - p] for r in range(a)], dtype=np.complex128)
    w1 = np.array([g[p - j0 - c] for c in range(e)], dtype=np.complex128)
    row0 = np.array([cross_entry(h, g, p, i0, j0 + c) for c in range(e)], dtype=np.complex128)
    col0 = np.array([cross_entry(h, g, p, i0 + r, j0) for r in range(a)], dtype=np.complex128)
    # R = D - v1 w1^T is supported on row 0 and column 0 only
    R0 = row0 - v1[0] * w1            # row 0 of R
    Rc = col0 - v1 * w1[0]            # column 0 of R
    e0 = np.zeros(a, dtype=np.complex128)
    e0[0] = 1
    u = Rc - Rc[0] * e0               # column 0 without its corner
    return [(v1, w1), (e0, R0), (u, np.eye(e, dtype=np.complex128)[0])]


def cross_apply(h, g, p, v, y1, chunk):
    """Compute z_i = sum_j E_ij y1_j for p <= i < v, chunk by chunk, via displacement products."""
    out = np.zeros(v - p, dtype=np.complex128)
    for i0 in range(p, v, chunk):
        a = min(chunk, v - i0)
        for j0 in range(0, p, chunk):
            e = min(chunk, p - j0)
            x = y1[j0:j0 + e]
            for (vv, ww) in chunk_displacement(h, g, p, i0, a, j0, e):
                out[i0 - p:i0 - p + a] += outer_apply(vv, ww, x, a)
    return out


def toeplitz_apply(h, x, chunk=4):
    """Apply the lower triangular Toeplitz T_v (v = len(x)) recursively to x."""
    x = np.asarray(x, dtype=np.complex128)
    v = len(x)
    if v == 1:
        return np.array([h[0] * x[0]])
    p = v // 2
    y1 = toeplitz_apply(h, x[:p], chunk)
    y2 = toeplitz_apply(h, x[p:], chunk)
    g = reciprocal_series(h, v)
    z = cross_apply(h, g, p, v, y1, chunk)
    return np.concatenate([y1, z + y2])
