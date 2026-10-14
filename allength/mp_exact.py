"""Step 8: the exact-width block Toeplitz apply in arbitrary precision (mpmath).

Purpose: separate algebraic correctness from float64 instability. The same construction as
allength/exact_width.py (convolution-DAG chunks, replay with borrowed coordinates, restored
exactly), but every scalar is an mpmath number, so the residual can be driven far below float64.

Only the chunk maps and the replay are re-implemented here; the DAG utilities (compose,
sum_dags) and replay_sequential are generic and reused unchanged.
"""

import mpmath as mp

from allength import conv_dag as conv_dag_module  # strict flag KEEP_ZERO_MULTIPLIERS
from allength.chunk_dag import compose, sum_dags

# ROOT_SIGN = -1 evaluates every power-of-two root as its inverse (= complex conjugate).
# Together with conjugated inputs (u -> 1/u) this is the preprint's preparation of conjugate
# coefficients: the same arithmetic DAG evaluated a second time with inverse roots.
ROOT_SIGN = 1
from allength.replay import DAG, replay_sequential


def mp_reciprocal(h, r):
    g = [None] * r
    g[0] = 1 / h[0]
    for k in range(1, r):
        g[k] = -(1 / h[0]) * sum(h[j] * g[k - j] for j in range(1, k + 1))
    return g


def mp_cross_entry(h, g, p, i, j):
    return sum(h[i - nu] * g[nu - j] for nu in range(j, p))


def _bitrev(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def mp_conv_dag(c, m):
    """DAG for x (length m) -> conv(x, c) (length m+len(c)-1), mp coefficients."""
    k = len(c)
    v = m + k - 1
    L = 1
    while L < v:
        L <<= 1
    bits = max(1, L.bit_length() - 1)
    # fixed transform of c (direct DFT, mp)
    cp = [c[t] if t < k else mp.mpc(0) for t in range(L)]
    Chat = [sum(cp[t] * mp.exp(-ROOT_SIGN * 2j * mp.pi * t * kk / L) for t in range(L))
            for kk in range(L)]

    gates = []

    def add_gate(preds):
        gates.append(preds)
        return ('z', len(gates) - 1)

    def stage(nodes, sign, size):
        half = size // 2
        out = list(nodes)
        for start in range(0, L, size):
            for j in range(half):
                p, q = start + j, start + j + half
                w = mp.exp(ROOT_SIGN * sign * 2j * mp.pi * j / size)
                a, b = nodes[p], nodes[q]
                pa = [] if a is None else [(a[0], a[1], mp.mpc(1))]
                pb = [] if b is None else [(b[0], b[1], w)]
                pm = [] if b is None else [(b[0], b[1], -w)]
                out[p] = add_gate(pa + pb) if (pa or pb) else None
                out[q] = add_gate(pa + pm) if (pa or pm) else None
        return out

    def fft(nodes, sign):
        nodes = [nodes[_bitrev(i, bits)] for i in range(L)]
        size = 2
        while size <= L:
            nodes = stage(nodes, sign, size)
            size <<= 1
        return nodes

    sources = [('x', t) if t < m else None for t in range(L)]
    spec = fft(sources, -1)
    scaled = []
    for kk in range(L):
        node = spec[kk]
        coef = Chat[kk] / L
        if node is None or (not conv_dag_module.KEEP_ZERO_MULTIPLIERS and coef == 0):
            scaled.append(None)
        else:
            scaled.append(add_gate([(node[0], node[1], coef)]))
    y = fft(scaled, +1)
    outs = []
    for t in range(v):
        node = y[t]
        outs.append(None if node is None else (node[0], node[1], mp.mpc(1)))
    return DAG(m, v, gates, outs)


def mp_outer_dag(v, w, e):
    a = len(v)
    d1 = mp_conv_dag(w, e)

    def relabel(kind, idx):
        return (e - 1 - idx) if kind == 'x' else idx

    gates1 = [[(k, relabel(k, i), c) for k, i, c in preds] for preds in d1.gates]
    sel = []
    for l in range(e):
        o = d1.outputs[e - 1 - l]
        sel.append(None if o is None else (o[0], relabel(o[0], o[1]), o[2]))
    d1b = DAG(e, e, gates1, sel)
    d2_full = mp_conv_dag(v, e)
    d2 = DAG(e, a, d2_full.gates, d2_full.outputs[:a])
    return compose(d1b, d2)


def mp_chunk_parts(h, g, p, i0, a, j0, e):
    v1 = [-h[i0 + r - p] for r in range(a)]
    w1 = [g[p - j0 - c] for c in range(e)]
    row0 = [mp_cross_entry(h, g, p, i0, j0 + c) for c in range(e)]
    col0 = [mp_cross_entry(h, g, p, i0 + r, j0) for r in range(a)]
    R0 = [row0[c] - v1[0] * w1[c] for c in range(e)]
    Rc = [col0[r] - v1[r] * w1[0] for r in range(a)]
    u = [Rc[r] - (Rc[0] if r == 0 else 0) for r in range(a)]
    e0 = [mp.mpc(1) if r == 0 else mp.mpc(0) for r in range(a)]
    ej = [mp.mpc(1) if c == 0 else mp.mpc(0) for c in range(e)]
    return [(v1, w1), (e0, R0), (u, ej)]


def mp_chunk_dag(parts, e):
    return sum_dags([mp_outer_dag(vv, ww, e) for vv, ww in parts])


def mp_block_apply(h, s0, chunk):
    """Exact-width Toeplitz apply in mp arithmetic; returns list of mpc on the r block coordinates."""
    s = [mp.mpc(x) for x in s0]
    r = len(s)
    g = mp_reciprocal(h, r + 1)

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            s[lo] = h[0] * s[lo]
            return
        p = v // 2
        rec(lo, lo + p)
        rec(lo + p, hi)
        for i0 in range(lo + p, hi, chunk):
            I = list(range(i0, min(i0 + chunk, hi)))
            for j0 in range(lo, lo + p, chunk):
                J = list(range(j0, min(j0 + chunk, lo + p)))
                parts = mp_chunk_parts(h, g, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = mp_chunk_dag(parts, len(J))
                used = set(I) | set(J)
                borrowed = [q for q in range(r) if q not in used]
                assert len(borrowed) >= dag.g, "chunk does not fit the block"
                Z = borrowed[:dag.g]
                w = [s[j] for j in J] + [s[i] for i in I] + [s[z] for z in Z]
                w = replay_sequential(dag, w)
                e_, a_ = len(J), len(I)
                for i, val in zip(I, w[e_:e_ + a_]):
                    s[i] = val
                for z, val in zip(Z, w[e_ + a_:]):
                    s[z] = val
    rec(0, r)
    return s


# ---- block-local word in mp arithmetic (step 10b) ---------------------------------------
from allength.rounds import admissible_chunk  # noqa: E402


def mp_block_apply_local(h, s0):
    """Block-local exact-width word (as in the paper): admissible chunk per block, borrowing
    only inside the block, direct fallback when no chunk is admissible. Returns list of mpc."""
    s = [mp.mpc(x) for x in s0]
    r = len(s)
    g = mp_reciprocal(h, r + 1)

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            s[lo] = h[0] * s[lo]
            return
        b = admissible_chunk(v)
        if b == 0:
            for i in range(v - 1, -1, -1):          # rows in decreasing order
                s[lo + i] = h[0] * s[lo + i]
                for j in range(i):
                    s[lo + i] = s[lo + i] + h[i - j] * s[lo + j]
            return
        p = v // 2
        rec(lo, lo + p)
        rec(lo + p, hi)
        for i0 in range(lo + p, hi, b):
            I = list(range(i0, min(i0 + b, hi)))
            for j0 in range(lo, lo + p, b):
                J = list(range(j0, min(j0 + b, lo + p)))
                parts = mp_chunk_parts(h, g, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = mp_chunk_dag(parts, len(J))
                used = set(I) | set(J)
                borrowed = [q for q in range(lo, hi) if q not in used]
                assert len(borrowed) >= dag.g, "pair does not fit its block"
                Z = borrowed[:dag.g]
                w = [s[j] for j in J] + [s[i] for i in I] + [s[z] for z in Z]
                w = replay_sequential(dag, w)
                e_, a_ = len(J), len(I)
                for i, val in zip(I, w[e_:e_ + a_]):
                    s[i] = val
                for z, val in zip(Z, w[e_ + a_:]):
                    s[z] = val
    rec(0, r)
    return s
