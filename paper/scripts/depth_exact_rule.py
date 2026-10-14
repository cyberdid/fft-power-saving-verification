"""Depth model with the exact admissible-chunk rule (article, sections 5.2 and 5.6).

Recurrence:  d(v) = d(v//2) + k^2 * L(b),  k = ceil((v//2) / b),
             b = admissible chunk for width v (largest b with g(b) + 2b <= v,
             first failure stops the search, as in rounds.admissible_chunk),
             L(b) = number of layers of one chunk pair (depth_recurrence.pair_layers).
Direct cost for v < 8: v(v+1)/2.

Prints d(2^e) / log2(2^e)^4 for e = 12..17 and checks d(512), d(1024) against the
measured ASAP depths reported in the article (1021838, 1668994).
"""
import contextlib
import io
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

STRICT = "--strict" in sys.argv[1:]   # strict DAGs (no zero multiplier dropped)

with contextlib.redirect_stdout(io.StringIO()):
    from allength import conv_dag as CD  # noqa: E402
    from allength import depth_recurrence as DR  # noqa: E402
CD.KEEP_ZERO_MULTIPLIERS = STRICT

G = pickle.load(open(os.path.join(
    ROOT, "data", "gate_count_table_b_le_1024" + ("_strict" if STRICT else "") + ".pkl"), "rb"))


def adm(v):
    best = 0
    for b in range(1, 1025):
        if b > v:
            break
        if G[b] + 2 * b <= v:
            best = b
        else:
            break
    return best


_L = {}


def L(b):
    if b not in _L:
        with contextlib.redirect_stdout(io.StringIO()):
            _L[b] = DR.pair_layers(b)
    return _L[b]


_memo = {}


def d(v):
    if v in _memo:
        return _memo[v]
    if v == 1:
        out = 1
    else:
        b = adm(v)
        if b == 0:
            out = v * (v + 1) // 2
        else:
            p = v // 2
            k = (p + b - 1) // b
            out = d(p) + k * k * L(b)
    _memo[v] = out
    return out


if __name__ == "__main__":
    mode = "strict" if STRICT else "default (zero multipliers dropped)"
    print(f"mode: {mode}")
    print(f"model: d(256)={d(256)} d(512)={d(512)} d(1024)={d(1024)} "
          f"(compare with measured ASAP depths of the same mode)")
    for width in (64,128,256,288,320,336,352,384,512,640,768,1024):
        print(f"width-model r={width} b={adm(width)} depth={d(width)}", flush=True)
    print("  r        b     d(r)          d/log2(r)^4")
    for e in range(12, 18):
        v = 2 ** e
        print(f"2^{e:<2d}  {adm(v):5d}  {d(v):12d}   {d(v) / math.log2(v) ** 4:8.2f}", flush=True)
