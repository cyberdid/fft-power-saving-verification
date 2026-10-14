import sys, math, time
sys.path.insert(0, __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..')))
import numpy as np
from allength import rounds as RD
from allength.exact_width import chunk_gate_count
from allength.chunk_dag import chunk_dag
from allength.toeplitz_disp import chunk_displacement, reciprocal_series


def admissible_bisect(v):
    lo, hi = 0, 1
    while hi <= v and chunk_gate_count(None, hi, hi) + 2 * hi <= v:
        lo, hi = hi, hi * 2
    hi = min(hi, v + 1)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if chunk_gate_count(None, mid, mid) + 2 * mid <= v:
            lo = mid
        else:
            hi = mid
    return lo


def pair_layers(b, seed=0):
    """Replay layer count of one representative pair (a = e = b), generic values."""
    rng = np.random.default_rng(seed)
    n = 2 * b + 2
    h = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    h[0] = abs(h[0]) + 1
    p = b
    g = reciprocal_series(h, p + 1)
    parts = chunk_displacement(h, g, p, p, b, 0, b)
    dag = chunk_dag([vv for vv, _ in parts], [ww for _, ww in parts], b)
    return len(RD.replay_layers(dag))


def depth(v, cache):
    if v == 1:
        return 1
    b = admissible_bisect(v)
    if b == 0:
        return v * (v + 1) // 2          # direct word: sequential rows (small v only)
    p = v // 2
    if b not in cache:
        cache[b] = pair_layers(b)
    k = (p + b - 1) // b
    cross = k * k * cache[b]             # pairs processed serially
    return depth(p, cache) + cross


cache = {}
rows = []
for kk in range(4, 17):
    v = 2 ** kk
    t = time.time()
    d = depth(v, cache)
    rows.append((v, d))
    print(f'v=2^{kk:2d}={v:6d}  depth={d:12d}  depth/log2(v)^4={d / math.log2(v) ** 4:10.2f}  ({time.time()-t:.1f}s)', flush=True)
