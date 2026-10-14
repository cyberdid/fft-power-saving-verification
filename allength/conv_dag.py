"""Step 4: convolution with a fixed vector as a linear arithmetic DAG
(local.tex, eq. loc:convolution-bounds).

For fixed c (length k) and variable x (length m), y = conv(x, c) has length v = m+k-1.
Pad to L = least power of two >= v. Build:  x -> FFT_L (radix-2 butterflies, DAG)
-> pointwise multiply by the fixed transform of c (scalars, folded with 1/L)
-> inverse FFT_L (radix-2 butterflies, DAG) -> first v outputs.

Every gate has at most two predecessors (butterfly a +- w*b, or a single scalar multiple).
Zero padding costs no gates (missing predecessors are simply absent).
Complex coefficients; the DAG reuses replay.DAG (evaluate / fan_out / depths).
"""

import cmath
import math
from contextlib import contextmanager

import numpy as np

from allength.replay import DAG

# Strict mode (local.tex): the topology depends only on the width. When True, a pointwise
# multiplier whose coefficient is exactly zero is kept as a gate instead of being dropped.
# Default False reproduces the earlier (value-dependent) measurements.
KEEP_ZERO_MULTIPLIERS = False


@contextmanager
def zero_multiplier_mode(keep):
    """Choose the legacy/strict builder mode for one synchronous construction."""
    global KEEP_ZERO_MULTIPLIERS
    previous = KEEP_ZERO_MULTIPLIERS
    KEEP_ZERO_MULTIPLIERS = bool(keep)
    try:
        yield
    finally:
        KEEP_ZERO_MULTIPLIERS = previous


def _next_pow2(n):
    L = 1
    while L < n:
        L <<= 1
    return L


def _bitrev(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def conv_dag(c, m):
    """DAG M: C^m -> C^v, M x = conv(x, c) with v = m + len(c) - 1."""
    c = np.asarray(c, dtype=np.complex128)
    k = len(c)
    v = m + k - 1
    L = _next_pow2(v)
    bits = max(1, L.bit_length() - 1)
    C_hat = np.fft.fft(np.concatenate([c, np.zeros(L - k, dtype=np.complex128)]))

    gates = []

    def add_gate(preds):
        gates.append(preds)
        return ('z', len(gates) - 1)

    def butterfly_stage(nodes, sign, size):
        """One radix-2 stage of a DIT FFT of the given sign (-1 forward, +1 inverse)."""
        half = size // 2
        out = list(nodes)
        for start in range(0, L, size):
            for j in range(half):
                p = start + j
                q = p + half
                w = cmath.exp(sign * 2j * math.pi * j / size)
                a, b = nodes[p], nodes[q]
                pa = [] if a is None else [(a[0], a[1], 1)]
                pb = [] if b is None else [(b[0], b[1], w)]
                pm = [] if b is None else [(b[0], b[1], -w)]
                # p' = a + w b ; q' = a - w b
                out[p] = add_gate(pa + pb) if (pa or pb) else None
                out[q] = add_gate(pa + pm) if (pa or pm) else None
        return out

    def fft(nodes, sign):
        # bit-reversal permutation is wiring only
        nodes = [nodes[_bitrev(i, bits)] for i in range(L)]
        size = 2
        while size <= L:
            nodes = butterfly_stage(nodes, sign, size)
            size <<= 1
        return nodes

    sources = [('x', t) if t < m else None for t in range(L)]
    spec = fft(sources, -1)
    # pointwise multiply by C_hat / L (inverse transform normalisation folded in)
    scaled = []
    for kk in range(L):
        node = spec[kk]
        coef = C_hat[kk] / L
        if node is None or (not KEEP_ZERO_MULTIPLIERS and abs(coef) == 0):
            scaled.append(None)
        else:
            scaled.append(add_gate([(node[0], node[1], coef)]))
    y_nodes = fft(scaled, +1)

    outputs = []
    for t in range(v):
        node = y_nodes[t]
        if node is None:
            outputs.append(None)
        elif node[0] == 'x':
            outputs.append(('x', node[1], 1))
        else:
            outputs.append(('z', node[1], 1))
    # DAG expects gate predecessors as (kind, idx, coef) with 'z' idx referring to gates.
    gates = [[(kind, idx, c_) for (kind, idx, c_) in preds] for preds in gates]
    return DAG(m, v, gates, outputs)


def gate_count(dag):
    return len(dag.gates)
