"""Step 9: round count of the exact-width Toeplitz word (local.tex, loc:toeplitz "O(1+log^4 r) mixed rounds").

Model: the word is a sequence of shear layers on the r block coordinates. Each layer is a set of
shears on disjoint coordinate pairs (one round if all of them run together). A layer is placed in
the earliest round after every earlier layer that touched one of its coordinates (ASAP over
coordinate dependencies). The resulting number of rounds is a measured parallel depth of this
word; it is an upper bound on the depth any scheduler could achieve for this word, and it is
conservative (any touch counts as a dependency).

Checks:
  * executing the word round by round gives the same result as the sequential word;
  * the layer list of each chunk replay is exactly what replay_layered executes.
"""

from collections import defaultdict

import numpy as np

from allength.chunk_dag import chunk_dag
from allength.replay import colour_layers
from allength.replay import DAG  # noqa: F401  (used by type readers)
from allength.replay import build_sweeps
from allength.toeplitz_disp import reciprocal_series, chunk_displacement


def replay_layers(dag):
    """Layers (lists of (src, dst, coef)) of the four-sweep replay, in execution order.
    Local coordinates: x = [0, e), y = [e, e+a), z = [e+a, e+a+g)."""
    e, a, g = dag.e, dag.a, dag.g
    depths = dag.depths()

    def depth_of(coord):
        if coord < e + a:
            return 0
        return depths[coord - e - a]

    def layered(shears):
        return colour_layers(shears, depth_of)

    fwd, outs = build_sweeps(dag, drop_sources=False)
    fl = layered(fwd)
    layers = []
    layers += fl
    layers.append(outs)                        # readout (one layer)
    layers += [[(src, dst, -c) for src, dst, c in L] for L in reversed(fl)]  # undo
    fwd2, outs2 = build_sweeps(dag, drop_sources=True)
    fl2 = layered(fwd2)
    layers += fl2
    layers.append([(s, d, -c) for s, d, c in outs2])
    layers += [[(src, dst, -c) for src, dst, c in L] for L in reversed(fl2)]
    return [L for L in layers if L]            # empty layers take no round


def block_word(h, r, chunk):
    """Layers of the exact-width Toeplitz word on coordinates 0..r-1, in program order.
    Each cross chunk (I target, J source) is replayed with borrowed coordinates from the block."""
    g_full = reciprocal_series(h, r)
    word = []
    counter = [0]

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            word.append([(None, lo, h[0])])   # scale of coordinate lo by h_0 (not a shear)
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
                free = [q for q in range(r) if q not in used]
                # rotate the borrowed window so that successive chunks use different coordinates
                start = (counter[0] * max(1, dag.g)) % max(1, len(free))
                counter[0] += 1
                rotated = free[start:] + free[:start]
                Z = rotated[:dag.g]
                to_global = list(J) + list(I) + list(Z)
                for L in replay_layers(dag):
                    word.append([(to_global[s], to_global[d], c) for s, d, c in L])
    rec(0, r)
    return word


def schedule_asap(word):
    """Assign each layer a round: earliest round after every previous layer sharing a coordinate."""
    last = defaultdict(int)     # coordinate -> round of last layer touching it
    rounds = []
    for L in word:
        coords = {c for s, d, _ in L for c in ((s,) if s is not None else ()) + (d,)}
        rnd = 1 + max((last[c] for c in coords), default=0)
        for c in coords:
            last[c] = rnd
        rounds.append(rnd)
    return rounds


def _apply_layer(s, L):
    for src, dst, c in L:
        if src is None:
            s[dst] = c * s[dst]          # scaling
        else:
            s[dst] = s[dst] + c * s[src]  # shear (src != dst)


def execute_by_rounds(s, word, rounds):
    """Apply the word round by round; layers inside one round touch disjoint coordinates."""
    s = np.array(s, dtype=np.complex128)
    order = sorted(range(len(word)), key=lambda k: rounds[k])   # stable: keeps program order
    for k in order:
        _apply_layer(s, word[k])
    return s


def execute_sequential(s, word):
    s = np.array(s, dtype=np.complex128)
    for L in word:
        _apply_layer(s, L)
    return s


# ---- block-local word, as in the paper (step 10) ----------------------------------------
from allength.exact_width import chunk_gate_count  # noqa: E402


def admissible_chunk(v):
    """Largest chunk b such that every pair of chunk size b fits the block: g(b) + 2b <= v.
    Returns 0 if none (then the direct word is used). Gate count from the printed DAG skeleton."""
    best = 0
    b = 1
    while b <= v:
        if chunk_gate_count(None, b, b) + 2 * b <= v:
            best = b
            b += 1
        else:
            break
    return best


def direct_word(h, lo, v):
    """Direct word for a block of width v: rows in decreasing order, scale then add lower rows."""
    layers = []
    for i in range(v - 1, -1, -1):
        layers.append([(None, lo + i, h[0])])
        for j in range(i):
            layers.append([(lo + j, lo + i, h[i - j])])
    return layers


def block_word_local(h, r):
    """Exact-width word where each block of width v uses chunk size admissible for v and
    borrows only coordinates inside its own block (as in the paper)."""
    g_full = reciprocal_series(h, r)
    word = []

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            word.append([(None, lo, h[0])])
            return
        b = admissible_chunk(v)
        if b == 0:
            word.extend(direct_word(h, lo, v))
            return
        p = v // 2
        rec(lo, lo + p)
        rec(lo + p, hi)
        for i0 in range(lo + p, hi, b):
            I = list(range(i0, min(i0 + b, hi)))
            for j0 in range(lo, lo + p, b):
                J = list(range(j0, min(j0 + b, lo + p)))
                parts = chunk_displacement(h, g_full, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = chunk_dag([vv for vv, _ in parts], [ww for _, ww in parts], len(J))
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
