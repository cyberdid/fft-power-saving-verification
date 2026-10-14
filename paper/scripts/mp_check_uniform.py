"""Exactness check of the adaptive block word in mp arithmetic (article, section 5.1).

Usage:  python mp_check_uniform.py R DPS
  R    block width (352, 640, 1024 were reported)
  DPS  mpmath working precision in decimal digits (300, 400, 600 were used)

Input distribution: real and imaginary parts of h and x are uniform on [-1, 1]
(this is what produced the reported numbers; the article states it in section 4.2).
Seed: random.Random(R).  Output: relative error ||y - T x|| / ||T x||.
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import mpmath as mp  # noqa: E402
from allength import mp_exact as M  # noqa: E402

r = int(sys.argv[1])
dps = int(sys.argv[2])
mp.mp.dps = dps

rnd = random.Random(r)


def rc():
    return mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))


h = [rc() for _ in range(r + 1)]
h[0] = abs(h[0]) + 1
x = [rc() for _ in range(r)]

t = time.time()
got = M.mp_block_apply_local(h, x)
want = [sum(h[i - j] * x[j] for j in range(i + 1)) for i in range(r)]
num = mp.sqrt(sum(abs(got[i] - want[i]) ** 2 for i in range(r)))
den = mp.sqrt(sum(abs(w) ** 2 for w in want))
print(f"adaptive word r={r} mp({dps}): rel err={mp.nstr(num / den, 3)} ({round(time.time() - t)}s)", flush=True)
