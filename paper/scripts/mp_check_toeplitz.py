"""Strict-mode check of the Toeplitz word with random h (article v6, table 3).

Usage:  python mp_check_toeplitz.py R DPS [NVEC]
Same inputs as mp_check_uniform.py (random.Random(R): h_0..h_R then x, uniform on [-1, 1],
h_0 = |h_0| + 1); further vectors are the next draws. The word is built once in strict mode
(mp_compiler.mp_block_word_local, no tolerance) and applied to NVEC vectors.
Output: max relative error ||y - T x|| / ||T x|| and digits lost.
"""
import os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import mpmath as mp
from allength import conv_dag as CD
CD.KEEP_ZERO_MULTIPLIERS = True
from allength import mp_compiler as MC
from allength.mp_exact import mp_reciprocal

R, DPS = int(sys.argv[1]), int(sys.argv[2])
NVEC = int(sys.argv[3]) if len(sys.argv) > 3 else 3
mp.mp.dps = DPS
rnd = random.Random(R)
rc = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
h = [rc() for _ in range(R + 1)]
h[0] = abs(h[0]) + 1
xs = [[rc() for _ in range(R)] for _ in range(NVEC)]
t0 = time.time()
word = MC.mp_block_word_local(h, R)
tb = time.time() - t0
errs = []
for x in xs:
    y = list(x)
    for L in word:
        for s, d, c in L:
            if s is None:
                y[d] = c * y[d]
            else:
                y[d] = y[d] + c * y[s]
    want = [sum(h[i - j] * x[j] for j in range(i + 1)) for i in range(R)]
    num = mp.sqrt(sum(abs(y[i] - want[i]) ** 2 for i in range(R)))
    den = mp.sqrt(sum(abs(w) ** 2 for w in want))
    errs.append(num / den)
mx = max(errs)
g = mp_reciprocal(h, R + 1)
print(f"toeplitz strict r={R} dps={DPS} vectors={NVEC}: max rel err={mp.nstr(mx, 3)} "
      f"min={mp.nstr(min(errs), 3)} digits lost={float(DPS + mp.log10(mx)):.1f} "
      f"log10 max|g|={float(max(mp.log10(abs(z)) for z in g)):.1f} "
      f"layers={len(word)} build={tb:.0f}s total={time.time() - t0:.0f}s", flush=True)
