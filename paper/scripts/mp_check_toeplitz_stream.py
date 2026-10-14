"""Streaming strict-mode check of the Toeplitz word with random h (article v6, table 3).

Same construction and inputs as mp_check_toeplitz.py, but each chunk-pair DAG is built,
replayed on all vectors at once and discarded, so memory stays small at r = 640, 1024.
Usage:  python mp_check_toeplitz_stream.py R DPS [NVEC]
"""
import os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import mpmath as mp
from allength import conv_dag as CD
CD.KEEP_ZERO_MULTIPLIERS = True
from allength.mp_exact import mp_chunk_parts, mp_chunk_dag, mp_reciprocal
from allength.rounds import admissible_chunk, replay_layers

R, DPS = int(sys.argv[1]), int(sys.argv[2])
NVEC = int(sys.argv[3]) if len(sys.argv) > 3 else 3
mp.mp.dps = DPS
rnd = random.Random(R)
rc = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
h = [rc() for _ in range(R + 1)]
h[0] = abs(h[0]) + 1
xs = [[rc() for _ in range(R)] for _ in range(NVEC)]
ys = [list(x) for x in xs]
t0 = time.time()
g = mp_reciprocal(h, R + 1)
stats = {"layers": 0}


def apply(L):
    stats["layers"] += 1
    for y in ys:
        for s, d, c in L:
            if s is None:
                y[d] = c * y[d]
            else:
                y[d] = y[d] + c * y[s]


def rec(lo, hi):
    v = hi - lo
    if v == 1:
        apply([(None, lo, h[0])])
        return
    b = admissible_chunk(v)
    if b == 0:
        for i in range(v - 1, -1, -1):
            apply([(None, lo + i, h[0])])
            for j in range(i):
                apply([(lo + j, lo + i, h[i - j])])
        return
    p = v // 2
    rec(lo, lo + p)
    rec(lo + p, hi)
    for i0 in range(lo + p, hi, b):
        I = list(range(i0, min(i0 + b, hi)))
        for j0 in range(lo, lo + p, b):
            J = list(range(j0, min(j0 + b, lo + p)))
            dag = mp_chunk_dag(mp_chunk_parts(h, g, p, i0 - lo, len(I), j0 - lo, len(J)), len(J))
            used = set(I) | set(J)
            Z = [q for q in range(lo, hi) if q not in used][:dag.g]
            tg = list(J) + list(I) + list(Z)
            for L in replay_layers(dag):
                apply([(tg[s] if s is not None else None, tg[d], c) for s, d, c in L])


rec(0, R)
errs = []
for x, y in zip(xs, ys):
    want = [sum(h[i - j] * x[j] for j in range(i + 1)) for i in range(R)]
    num = mp.sqrt(sum(abs(y[i] - want[i]) ** 2 for i in range(R)))
    den = mp.sqrt(sum(abs(w) ** 2 for w in want))
    errs.append(num / den)
mx = max(errs)
print(f"toeplitz strict r={R} dps={DPS} vectors={NVEC}: max rel err={mp.nstr(mx, 3)} "
      f"min={mp.nstr(min(errs), 3)} digits lost={float(DPS + mp.log10(mx)):.1f} "
      f"log10 max|g|={float(max(mp.log10(abs(z)) for z in g)):.1f} "
      f"layers={stats['layers']} total={time.time() - t0:.0f}s", flush=True)
