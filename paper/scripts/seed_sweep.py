"""Seed sweep for the figures: ASAP depth, layers and operations of the block word
at several widths and seeds, for the adaptive rule, chunk size 1 and cap 2.

Usage:  python seed_sweep.py SEED [SEED ...]   (default seeds 0 1 2 3 4)
Input per seed k and width r:  default_rng(10000 + 1000*k + r), h0 = |h0|+1, complex normal.
Appends rows to paper/data/seed_sweep.csv:
  seed,r,rule,b_adm,depth,layers,ops
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from allength import rounds as RD  # noqa: E402

WIDTHS = [256, 288, 320, 336, 352, 384, 512, 640, 768, 1024]
OUT = os.path.join(ROOT, "paper", "data", "seed_sweep.csv")
orig = RD.admissible_chunk
RULES = {
    "adaptive": orig,
    "chunk1": lambda v: 1 if v >= 8 else 0,
    "cap2": lambda v: min(orig(v), 2) if orig(v) > 0 else 0,
}


def gen_h(seed, r):
    rng = np.random.default_rng(10000 + 1000 * seed + r)
    h = rng.standard_normal(r + 1) + 1j * rng.standard_normal(r + 1)
    h[0] = abs(h[0]) + 1
    return h


def run(seed, r, rule_name):
    RD.admissible_chunk = RULES[rule_name]
    try:
        word = RD.block_word_local(gen_h(seed, r), r)
        depth = max(RD.schedule_asap(word))
        ops = sum(len(L) for L in word)
        return depth, len(word), ops
    finally:
        RD.admissible_chunk = orig


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    new = not os.path.exists(OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["seed", "r", "rule", "b_adm", "depth", "layers", "ops"])
        for seed in seeds:
            for r in WIDTHS:
                for rule in ["adaptive", "chunk1"] + (["cap2"] if r == 512 else []):
                    t = time.time()
                    depth, layers, ops = run(seed, r, rule)
                    badm = orig(r) if rule != "chunk1" else 1
                    if rule == "cap2":
                        badm = min(orig(r), 2)
                    w.writerow([seed, r, rule, badm, depth, layers, ops])
                    f.flush()
                    print(f"seed={seed} r={r} {rule}: depth={depth} layers={layers} ops={ops} ({time.time()-t:.0f}s)", flush=True)
