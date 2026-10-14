"""Measured ASAP depth, layer count and scalar-operation count of the block word
on generic inputs (article, tables 2, 4, 5, 6).

Usage:  python asap_tables.py [r ...]        (default: 256 512)
Generic input: h0 = |h0| + 1, other h_k complex standard normal, seed = r.
Columns, per width r:
  adaptive : admissible chunk rule (rounds.admissible_chunk)
  chunk1   : chunk size fixed to 1 where it fits (v >= 8), direct otherwise
  cap2     : admissible rule capped at chunk size 2 (table 5 middle row)
Operations = sum over layers of the number of gates (each shear or scale = 1).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from allength import rounds as RD  # noqa: E402


def generic_h(r):
    rng = np.random.default_rng(r)
    h = rng.standard_normal(r + 1) + 1j * rng.standard_normal(r + 1)
    h[0] = abs(h[0]) + 1
    return h


def measure(r, rule):
    orig = RD.admissible_chunk
    RD.admissible_chunk = rule
    try:
        h = generic_h(r)
        word = RD.block_word_local(h, r)
        depth = max(RD.schedule_asap(word))
        ops = sum(len(L) for L in word)
        return depth, len(word), ops
    finally:
        RD.admissible_chunk = orig


if __name__ == "__main__":
    widths = [int(a) for a in sys.argv[1:]] or [256, 512]
    orig = RD.admissible_chunk
    chunk1 = lambda v: 1 if v >= 8 else 0  # noqa: E731
    cap2 = lambda v: min(orig(v), 2) if orig(v) > 0 else 0  # noqa: E731
    print(f"{'r':>5} {'b_adm':>5} | {'adapt depth':>12} {'layers':>10} {'ops':>11} | "
          f"{'chunk1 depth':>13} {'layers':>10} {'ops':>11} | {'cap2 depth':>11} {'ops':>11}")
    for r in widths:
        ad, al, ao = measure(r, orig)
        cd, cl, co = measure(r, chunk1)
        pd, _, po = measure(r, cap2)
        print(f"{r:>5} {orig(r):>5} | {ad:>12} {al:>10} {ao:>11} | "
              f"{cd:>13} {cl:>10} {co:>11} | {pd:>11} {po:>11}", flush=True)
