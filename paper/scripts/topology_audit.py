"""Fixed-topology audit of the strict word (article v6, table 4).

Usage:  python topology_audit.py OUT.csv R [R ...]
For each width R, both rules (adaptive, chunk1) and three inputs (seed = R, another seed,
h = ones) the word is streamed and a SHA-256 is taken over, per layer, the ASAP round and every
gate as (type, src, dst) with type M (monomial/scale) or S (shear). Equal hashes for the three
inputs mean equal gate types, pairs, layer order and round assignment.
"""
import csv, hashlib, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strict_tables as ST  # noqa: E402  (sets strict mode)


class Sink2(ST.Sink):
    def __init__(self):
        super().__init__()
        self.h2 = hashlib.sha256()

    def add(self, L):
        if not L:
            return
        before = dict()
        coords = set()
        for s, d, _ in L:
            if s is not None:
                coords.add(s)
            coords.add(d)
        rnd = 1 + max(self.last[c] for c in coords)
        super().add(L)
        self.h2.update((f"{rnd}:" + ";".join(("M" if s is None else "S") + f"{s},{d}"
                                               for s, d, _ in L) + "|").encode())


out = sys.argv[1]
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["r", "rule", "input", "depth", "layers", "shears", "scales", "hash_types_rounds"])
    for R in map(int, sys.argv[2:]):
        for rule in ("adaptive", "chunk1"):
            for inp in ("seed", "seed2", "ones"):
                t = time.time()
                sk = Sink2()
                ST.stream_word(ST.make_h(R, inp), R, ST.RULES[rule], sk)
                row = [R, rule, inp, sk.depth, sk.layers, sk.shears, sk.scales, sk.h2.hexdigest()[:20]]
                w.writerow(row); f.flush()
                print(*row, f"{time.time()-t:.0f}s", flush=True)
