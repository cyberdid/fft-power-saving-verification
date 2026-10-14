"""Step 6: exact-width Toeplitz apply inside one block of r coordinates (local.tex, loc:toeplitz).

The whole transform acts on the r coordinates of the block s. Cross updates are done chunk by
chunk. For a chunk with target coordinates I and source coordinates J, the chunk map
    s[I] += E[I,J] s[J]
is realised by the replay lemma (loc:replay): a linear DAG for the chunk is run with its
gate coordinates BORROWED from other coordinates of the same block (any contents are allowed),
so no workspace outside the block is used. The borrowed coordinates are restored exactly.

Simplifications, stated honestly:
  * the chunk DAG is a dense DAG (one scalar gate per nonzero entry plus a sum chain); its gate
    count is about 2ab, not the O(v log v) convolution DAG of the paper. Therefore the round
    count and the logarithmic bounds are NOT verified here.
  * replay runs on a gathered copy [x-chunk | y-chunk | borrowed] and results are scattered back;
    this is the same action on the block coordinates, because the replay is a fixed word on
    its own coordinates.
"""

import numpy as np

from allength.replay import DAG, replay_sequential
from allength.toeplitz_disp import reciprocal_series, cross_entry


def dense_dag(M):
    """Linear DAG for a dense complex matrix M (a x e): sums of scalar gates."""
    a, e = M.shape
    gates, outputs = [], []
    for i in range(a):
        terms = []
        for j in range(e):
            if M[i, j] != 0:
                gates.append([('x', j, complex(M[i, j]))])
                terms.append(('z', len(gates) - 1))
        if not terms:
            outputs.append(None)
            continue
        acc = terms[0]
        for t in terms[1:]:
            gates.append([(acc[0], acc[1], 1), (t[0], t[1], 1)])
            acc = ('z', len(gates) - 1)
        outputs.append((acc[0], acc[1], 1))
    return DAG(e, a, gates, outputs)


def chunk_update(s, I, J, Mchunk, borrowed):
    """Apply s[I] += Mchunk @ s[J] by replay with borrowed coordinates (restored exactly)."""
    dag = dense_dag(Mchunk)
    assert len(borrowed) >= dag.g, "not enough borrowed coordinates in the block"
    Z = borrowed[:dag.g]
    w = [s[j] for j in J] + [s[i] for i in I] + [s[z] for z in Z]
    w = replay_sequential(dag, w)
    e, a = len(J), len(I)
    x_back = w[:e]
    y_new = w[e:e + a]
    z_back = w[e + a:]
    for j, val in zip(J, x_back):
        assert abs(s[j] - val) < 1e-9 * max(1, abs(val))
    for i, val in zip(I, y_new):
        s[i] = val
    for z, val in zip(Z, z_back):
        s[z] = val  # restored: must equal the original contents (checked by the caller)


def block_toeplitz_apply(h, s0, chunk, check=True):
    """Apply lower triangular Toeplitz T_r (r = len(s0)) to s0, in place on r coordinates."""
    s = np.array(s0, dtype=np.complex128)
    r = len(s)
    g_full = reciprocal_series(h, r)

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
                Mchunk = np.array([[cross_entry(h, g_full, p, i - lo, j - lo) for j in J] for i in I])
                used = set(I) | set(J)
                borrowed = [q for q in range(r) if q not in used]
                before = s.copy()
                chunk_update(s, I, J, Mchunk, borrowed)
                if check:
                    gz = dense_dag(Mchunk).g
                    Z = borrowed[:gz]
                    assert np.allclose(s[Z], before[Z], atol=1e-9)
    rec(0, r)
    return s


def max_chunk_for_width(r):
    """Largest chunk b with dense-DAG gate count g(b) <= r - 2b (all borrowed fit in the block)."""
    best = 0
    for b in range(1, r):
        g = b * b + b * (b - 1)  # scalar gates b*b, sum chain b*(b-1)
        if g <= r - 2 * b:
            best = b
    return best


# ---- convolution-DAG chunks (step 7) ---------------------------------------
from allength.chunk_dag import chunk_dag  # noqa: E402
from allength.toeplitz_disp import chunk_displacement  # noqa: E402


def chunk_update_dag(s, I, J, dag, borrowed):
    """s[I] += dag(s[J]) by replay with borrowed coordinates restored exactly."""
    assert len(borrowed) >= dag.g, f"needs {dag.g} borrowed, has {len(borrowed)}"
    Z = borrowed[:dag.g]
    e, a = len(J), len(I)
    w = [s[j] for j in J] + [s[i] for i in I] + [s[z] for z in Z]
    w = replay_sequential(dag, w)
    for i, val in zip(I, w[e:e + a]):
        s[i] = val
    for z, val in zip(Z, w[e + a:]):
        s[z] = val
    return Z


def block_toeplitz_apply_conv(h, s0, chunk, check=True):
    """Exact-width Toeplitz apply with convolution-DAG chunks (no workspace outside the block)."""
    s = np.array(s0, dtype=np.complex128)
    r = len(s)
    g_full = reciprocal_series(h, r)

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
                parts = chunk_displacement(h, g_full, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = chunk_dag([vv for vv, _ in parts], [ww for _, ww in parts], len(J))
                used = set(I) | set(J)
                borrowed = [q for q in range(r) if q not in used]
                before = s.copy()
                Z = chunk_update_dag(s, I, J, dag, borrowed)
                if check:
                    assert np.allclose(s[Z], before[Z], atol=1e-8)
    rec(0, r)
    return s


def chunk_gate_count(h, a, e, rng_seed=0):
    """Gate count of the convolution-DAG chunk for an a x e block at generic values."""
    import numpy as _np
    rng = _np.random.default_rng(rng_seed)
    hh = rng.standard_normal(2 * (a + e) + 2) + 1j * rng.standard_normal(2 * (a + e) + 2)
    hh[0] = abs(hh[0]) + 1.0
    p = a + e
    g = reciprocal_series(hh, p + 1)
    parts = chunk_displacement(hh, g, p, p, a, 0, e)
    return chunk_dag([vv for vv, _ in parts], [ww for _, ww in parts], e).g


def max_chunk_conv(r):
    """Largest b with (gates of conv chunk a=e=b) <= r - 2b."""
    best = 0
    for b in range(1, r):
        if chunk_gate_count(None, b, b) <= r - 2 * b:
            best = b
        else:
            break
    return best
