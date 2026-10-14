"""Measured ASAP depth and layer count of the chunk-size-1 word at r = 1024 (article, table 3).

Generic input: h0 = |h0| + 1, other h_k complex standard normal, seed = r.
Reported: layers 10445232, ASAP depth 6291256 (about 70 s, several GB of RAM).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from allength import rounds as RD  # noqa: E402

r = 1024
rng = np.random.default_rng(r)
h = rng.standard_normal(r + 1) + 1j * rng.standard_normal(r + 1)
h[0] = abs(h[0]) + 1
RD.admissible_chunk = lambda v: 1 if v >= 8 else 0

t = time.time()
w = RD.block_word_local(h, r)
print("layers", len(w), "built", round(time.time() - t), "s", flush=True)
print("ASAP depth", max(RD.schedule_asap(w)), "total", round(time.time() - t), "s", flush=True)
