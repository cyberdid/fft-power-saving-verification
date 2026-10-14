"""Strict mode: round-by-round execution equals sequential execution; no coordinate conflicts
inside a round (article v6, section 6.9).  Usage: python rounds_check.py"""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np
from allength import conv_dag as CD
CD.KEEP_ZERO_MULTIPLIERS = True
from allength import rounds as RD


def gh(r):
    rng = np.random.default_rng(r)
    h = rng.standard_normal(r + 1) + 1j * rng.standard_normal(r + 1)
    h[0] = abs(h[0]) + 1
    return h


for r in (16, 32, 48, 64):
    w = RD.block_word_local(gh(r), r)
    rounds = RD.schedule_asap(w)
    x = np.random.default_rng(7).standard_normal(r) + 0j
    d = np.max(np.abs(RD.execute_by_rounds(x, w, rounds) - RD.execute_sequential(x, w)))
    print(f"r={r}: max |by rounds - sequential| = {d}", flush=True)
for r in (256, 512):
    w = RD.block_word_local(gh(r), r)
    rounds = RD.schedule_asap(w)
    seen = defaultdict(set)
    conflicts = 0
    for L, k in zip(w, rounds):
        coords = {c for s, dd, _ in L for c in ((s,) if s is not None else ()) + (dd,)}
        if coords & seen[k]:
            conflicts += 1
        seen[k] |= coords
    per = len(w) / max(rounds)
    print(f"r={r}: layers={len(w)} rounds={max(rounds)} conflicts={conflicts} layers/round={per:.2f}", flush=True)
