"""The strict block word over an arbitrary exact field (article v6, section 6.1b).

Same construction as mp_exact / mp_compiler.mp_block_word_local (reciprocal series, the three
outer products of the chunk displacement, convolution DAGs through power-of-two transforms,
four-sweep replay with borrowed coordinates), with every scalar in a field F from
allength.exact_field (one, zero, root(L, k), + - * /). Strict mode: no multiplier is dropped.
Only structural choices (admissible chunk, layer colouring) are taken from the float code;
they do not depend on coefficient values.
"""
from allength.chunk_dag import compose, sum_dags
from allength.replay import DAG
from allength.rounds import admissible_chunk, replay_layers


def g_reciprocal(F, h, r):
    g = [None] * r
    inv0 = F.one() / h[0]
    g[0] = inv0
    for k in range(1, r):
        acc = F.zero()
        for j in range(1, k + 1):
            acc = acc + h[j] * g[k - j]
        g[k] = -(inv0 * acc)
    return g


def g_cross_entry(F, h, g, p, i, j):
    acc = F.zero()
    for nu in range(j, p):
        acc = acc + h[i - nu] * g[nu - j]
    return acc


def _bitrev(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def g_conv_dag(F, c, m):
    k = len(c)
    v = m + k - 1
    L = 1
    while L < v:
        L <<= 1
    bits = max(1, L.bit_length() - 1)
    cp = [c[t] if t < k else F.zero() for t in range(L)]
    Chat = []
    for kk in range(L):
        acc = F.zero()
        for t in range(L):
            acc = acc + cp[t] * F.root(L, -t * kk)
        Chat.append(acc)
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
                w = F.root(size, sign * j)
                a, b = nodes[p], nodes[q]
                pa = [] if a is None else [(a[0], a[1], F.one())]
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
        if node is None:
            scaled.append(None)
        else:                                   # strict: kept even if the coefficient is 0
            scaled.append(add_gate([(node[0], node[1], Chat[kk] / L)]))
    y = fft(scaled, +1)
    outs = [None if y[t] is None else (y[t][0], y[t][1], F.one()) for t in range(v)]
    return DAG(m, v, gates, outs)


def g_outer_dag(F, v, w, e):
    a = len(v)
    d1 = g_conv_dag(F, w, e)

    def relabel(kind, idx):
        return (e - 1 - idx) if kind == 'x' else idx

    gates1 = [[(k, relabel(k, i), c) for k, i, c in preds] for preds in d1.gates]
    sel = []
    for l in range(e):
        o = d1.outputs[e - 1 - l]
        sel.append(None if o is None else (o[0], relabel(o[0], o[1]), o[2]))
    d1b = DAG(e, e, gates1, sel)
    d2_full = g_conv_dag(F, v, e)
    d2 = DAG(e, a, d2_full.gates, d2_full.outputs[:a])
    return compose(d1b, d2)


def g_chunk_parts(F, h, g, p, i0, a, j0, e):
    v1 = [-h[i0 + r - p] for r in range(a)]
    w1 = [g[p - j0 - c] for c in range(e)]
    row0 = [g_cross_entry(F, h, g, p, i0, j0 + c) for c in range(e)]
    col0 = [g_cross_entry(F, h, g, p, i0 + r, j0) for r in range(a)]
    R0 = [row0[c] - v1[0] * w1[c] for c in range(e)]
    Rc = [col0[r] - v1[r] * w1[0] for r in range(a)]
    u = [Rc[r] - (Rc[0] if r == 0 else F.zero()) for r in range(a)]
    e0 = [F.one() if r == 0 else F.zero() for r in range(a)]
    ej = [F.one() if c == 0 else F.zero() for c in range(e)]
    return [(v1, w1), (e0, R0), (u, ej)]


def g_block_word_local(F, h, r):
    g = g_reciprocal(F, h, r + 1)
    word = []

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            word.append([(None, lo, h[0])])
            return
        b = admissible_chunk(v)
        if b == 0:
            for i in range(v - 1, -1, -1):
                word.append([(None, lo + i, h[0])])
                for j in range(i):
                    word.append([(lo + j, lo + i, h[i - j])])
            return
        p = v // 2
        rec(lo, lo + p)
        rec(lo + p, hi)
        for i0 in range(lo + p, hi, b):
            I = list(range(i0, min(i0 + b, hi)))
            for j0 in range(lo, lo + p, b):
                J = list(range(j0, min(j0 + b, lo + p)))
                parts = g_chunk_parts(F, h, g, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = sum_dags([g_outer_dag(F, vv, ww, len(J)) for vv, ww in parts])
                used = set(I) | set(J)
                borrowed = [q for q in range(lo, hi) if q not in used]
                assert len(borrowed) >= dag.g, "pair does not fit its block"
                Z = borrowed[:dag.g]
                to_global = list(J) + list(I) + list(Z)
                for L in replay_layers(dag):
                    word.append([(to_global[s] if s is not None else None, to_global[d], c)
                                 for s, d, c in L])
    rec(0, r)
    return word
