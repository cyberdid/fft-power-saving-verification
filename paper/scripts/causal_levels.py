"""Decomposition of the Toeplitz-word intermediate-value GROWTH by recursion level and
phase (article v6.1, table 2c; sections 6.2 and 8.7).

Scope.  This decomposes the growth of the running peak |coordinate|, NOT the rounding
error itself.  For T^T the stage-wise analysis (table 2b) shows local loss ~= log10 growth,
so the growth decomposition is strong indirect evidence for where the T^T error comes from;
for T the loss exceeds the growth (operator amplification), so there the table describes
growth only.  Isolating each phase's contribution to the FINAL error (with error transport
and possible cancellation) would need a separate experiment and is not done here.

The block word is built by rec(lo, hi): a block of width v = hi-lo either is a leaf
(v == 1), a direct word (no admissible chunk), or splits into halves plus an inter-block
chunk phase that replays displacement-rank DAGs built from the reciprocal series g.  We
tag every emitted layer with (v, phase) and then replay the whole word on a high-precision
reference vector, tracking the running peak P.  Each layer's positive increment
  max(0, log10 P_after - log10 P_before)
is charged to that layer's (v, phase).  These increments sum exactly to the total
log10(global peak / ||input||_inf) (a telescoping identity), so the table is an additive
decomposition of the growth across recursion levels and phases.

Usage:  python causal_levels.py R DPS [--tword-only|--word-only]
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import mpmath as mp  # noqa: E402

from allength import conv_dag as CD  # noqa: E402

from allength.mp_exact import mp_chunk_parts, mp_chunk_dag, mp_reciprocal  # noqa: E402
from allength.rounds import admissible_chunk, replay_layers  # noqa: E402
from allength.compiler import transpose_word  # noqa: E402


def tagged_block_word(h, r):
    """Like mp_block_word_local but returns (word, tags); tags[i] = (v, phase) for layer i.
    phase in {'leaf','direct','chunk'}.  Transpose preserves the per-layer tag order
    reversed, handled by the caller."""
    g = mp_reciprocal(h, r + 1)
    word, tags = [], []

    def emit(layer, v, phase):
        word.append(layer)
        tags.append((v, phase))

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            emit([(None, lo, h[0])], v, 'leaf')
            return
        b = admissible_chunk(v)
        if b == 0:
            for i in range(v - 1, -1, -1):
                emit([(None, lo + i, h[0])], v, 'direct')
                for j in range(i):
                    emit([(lo + j, lo + i, h[i - j])], v, 'direct')
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
                Z = borrowed[:dag.g]
                to_global = list(J) + list(I) + list(Z)
                for L in replay_layers(dag):
                    emit([(to_global[s] if s is not None else None, to_global[d], c)
                          for s, d, c in L], v, 'chunk')

    rec(0, r)
    return word, tags


def replay_charge(word, tags, y):
    """Replay word on y, charging positive log10 peak increments to (v, phase).
    Returns dict (v,phase)->charge, the input inf-norm, and the global peak."""
    y = list(y)
    peak = max(abs(v) for v in y)
    in_inf = peak
    charge = {}
    for layer, tag in zip(word, tags):
        before = peak
        for s, d, c in layer:
            if s is None:
                y[d] = c * y[d]
            else:
                y[d] = y[d] + c * y[s]
            a = abs(y[d])
            if a > peak:
                peak = a
        if peak > before:
            charge[tag] = charge.get(tag, mp.mpf(0)) + (mp.log10(peak) - mp.log10(before))
    return charge, in_inf, peak, y


if __name__ == "__main__":
    CD.KEEP_ZERO_MULTIPLIERS = True
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    R, DPS = int(args[0]), int(args[1])
    mp.mp.dps = DPS + 150
    u = mp.exp(2j * mp.pi / R)
    H = [mp.mpc(1)]
    for j in range(1, R + 1):
        H.append(H[-1] * (1 - u ** j))
    cc = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(R)]
    h = [1 / H[j] for j in range(R)] + [mp.mpc(0)]
    Dh = H[:R]
    Dmid = [cc[k] / H[k] for k in range(R)]

    word, tags = tagged_block_word(h, R)
    # transpose reverses layer order and swaps shear ends; tag follows its layer.
    tword = transpose_word(word)
    ttags = list(reversed(tags))

    rnd = random.Random(R)
    x = [mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1)) for _ in range(R)]

    print(f"r={R} dps={DPS}  (charges are log10 of running-peak growth, additive)")
    print(f"total layers={len(word)}  recursion levels (block widths v): "
          f"{sorted(set(v for v, _ in tags))}")

    stages = []
    if "--word-only" not in sys.argv:
        stages.append(("T^T", tword, ttags, [Dh[i] * x[i] for i in range(R)]))
    if "--tword-only" not in sys.argv:
        # feed T the output of Dh T^T Dmid so its input matches the real chain
        yA = [Dh[i] * x[i] for i in range(R)]
        _, _, _, yA = replay_charge(tword, ttags, yA)
        yB = [Dmid[i] * yA[i] for i in range(R)]
        stages.append(("T", word, tags, yB))

    for name, w, tg, yin in stages:
        charge, in_inf, peak, _ = replay_charge(w, tg, yin)
        total = mp.log10(peak / in_inf)
        print(f"\nstage {name}: log10(peak/input_inf) = {float(total):.2f}")
        by_v = {}
        by_phase = {}
        for (v, ph), c in charge.items():
            by_v[v] = by_v.get(v, mp.mpf(0)) + c
            by_phase[ph] = by_phase.get(ph, mp.mpf(0)) + c
        print("  by recursion level v:")
        for v in sorted(by_v, reverse=True):
            print(f"    v={v:<5} growth={float(by_v[v]):8.2f}  "
                  f"({float(by_v[v] / total * 100):5.1f}% of total)")
        print("  by phase:")
        for ph in ('chunk', 'direct', 'leaf'):
            if ph in by_phase:
                print(f"    {ph:<7} growth={float(by_phase[ph]):8.2f}  "
                      f"({float(by_phase[ph] / total * 100):5.1f}%)")
