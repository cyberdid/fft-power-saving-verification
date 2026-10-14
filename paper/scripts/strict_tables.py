"""Strict-mode measurements of the block word (article v5, tables 2s, 4s, 5s).

Strict mode = conv_dag.KEEP_ZERO_MULTIPLIERS = True: no gate is dropped because its
coefficient happens to be zero, so the topology of the word depends only on the width
(as required by local.tex). The word is streamed into an online ASAP scheduler, so even
r = 1024 does not need the whole word in memory.

Usage:  python strict_tables.py OUT.csv r[:rule] ...
  rule in {adaptive, chunk1, cap2}; default: all three for r = 512, adaptive+chunk1 otherwise.
Inputs: h from default_rng(r) (h0 = |h0| + 1, complex normal), plus a structure check:
  for every task the structural signature (src, dst per layer, in order) is hashed; run the
  same task with --input ones / --input seed2 to compare signatures.
Columns: r,rule,input,b_adm,depth,layers,shears,scales,ops,signature,seconds
"""
import csv
import hashlib
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from allength import conv_dag as CD  # noqa: E402
from allength import rounds as RD  # noqa: E402
from allength.chunk_dag import chunk_dag  # noqa: E402
from allength.toeplitz_disp import reciprocal_series, chunk_displacement  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True
ORIG = RD.admissible_chunk
from allength.exact_width import chunk_gate_count  # noqa: E402

# In strict mode g(1) = 8 (6 with zero pruning), so chunk size 1 fits only when g(1) + 2 <= v,
# i.e. v >= 10 (the non-strict rule used v >= 8).
G1 = chunk_gate_count(None, 1, 1)
RULES = {
    "adaptive": ORIG,
    "chunk1": lambda v: 1 if G1 + 2 <= v else 0,
    "cap2": lambda v: min(ORIG(v), 2) if ORIG(v) > 0 else 0,
}


def make_h(r, kind):
    if kind == "ones":
        return np.ones(r + 1, dtype=np.complex128)
    seed = r if kind == "seed" else 10**6 + r
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(r + 1) + 1j * rng.standard_normal(r + 1)
    h[0] = abs(h[0]) + 1
    return h


class Sink:
    """Online ASAP schedule + counters + structural hash for a stream of layers."""

    def __init__(self):
        self.last = defaultdict(int)
        self.depth = 0
        self.layers = 0
        self.shears = 0
        self.scales = 0
        self.hash = hashlib.sha256()

    def add(self, L):
        if not L:
            return
        coords = set()
        parts = []
        for s, d, _ in L:
            if s is None:
                self.scales += 1
            else:
                self.shears += 1
                coords.add(s)
            coords.add(d)
            parts.append(f"{s},{d}")
        rnd = 1 + max(self.last[c] for c in coords)
        for c in coords:
            self.last[c] = rnd
        if rnd > self.depth:
            self.depth = rnd
        self.layers += 1
        self.hash.update((";".join(parts) + "|").encode())


def stream_word(h, r, rule, sink):
    """Same construction as rounds.block_word_local, layers sent to sink instead of a list."""
    g_full = reciprocal_series(h, r)

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            sink.add([(None, lo, h[0])])
            return
        b = rule(v)
        if b == 0:
            for L in RD.direct_word(h, lo, v):
                sink.add(L)
            return
        p = v // 2
        rec(lo, lo + p)
        rec(lo + p, hi)
        for i0 in range(lo + p, hi, b):
            I = list(range(i0, min(i0 + b, hi)))
            for j0 in range(lo, lo + p, b):
                J = list(range(j0, min(j0 + b, lo + p)))
                parts = chunk_displacement(h, g_full, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = chunk_dag([vv for vv, _ in parts], [ww for _, ww in parts], len(J))
                used = set(I) | set(J)
                borrowed = [q for q in range(lo, hi) if q not in used]
                assert len(borrowed) >= dag.g, "pair does not fit its block"
                Z = borrowed[:dag.g]
                to_global = list(J) + list(I) + list(Z)
                for L in RD.replay_layers(dag):
                    sink.add([(to_global[s] if s is not None else None, to_global[d], c)
                              for s, d, c in L])
    rec(0, r)


def run(r, rule_name, inp):
    t = time.time()
    sink = Sink()
    rule = RULES[rule_name]
    RD.admissible_chunk = rule          # rounds.direct_word does not use it; kept for symmetry
    try:
        stream_word(make_h(r, inp), r, rule, sink)
    finally:
        RD.admissible_chunk = ORIG
    return [r, rule_name, inp, rule(r), sink.depth, sink.layers, sink.shears, sink.scales,
            sink.shears + sink.scales, sink.hash.hexdigest()[:16], round(time.time() - t)]


if __name__ == "__main__":
    out = sys.argv[1]
    args = sys.argv[2:]
    inp = "seed"
    if "--input" in args:
        k = args.index("--input")
        inp = args[k + 1]
        args = args[:k] + args[k + 2:]
    tasks = []
    for a in args:
        if ":" in a:
            r, rule = a.split(":")
            tasks.append((int(r), rule))
        else:
            r = int(a)
            tasks += [(r, "adaptive"), (r, "chunk1")] + ([(r, "cap2")] if r == 512 else [])
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["r", "rule", "input", "b_adm", "depth", "layers", "shears", "scales",
                        "ops", "signature", "seconds"])
        for r, rule in tasks:
            row = run(r, rule, inp)
            w.writerow(row)
            f.flush()
            print(*row, flush=True)
