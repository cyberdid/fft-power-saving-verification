"""Build the exact gate-count table g(b) = gates of the convolution-DAG chunk of size b x b.

Output: data/gate_count_table_b_le_1024.pkl (dict b -> g(b), b = 1..1024).
Runtime on the reference machine: about 850 s (the DAG for large b is built explicitly).
"""
import os
import pickle
import sys
import contextlib
import io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

STRICT = "--strict" in sys.argv[1:]   # keep zero multipliers (topology depends only on b)

with contextlib.redirect_stdout(io.StringIO()):
    from allength import conv_dag as CD  # noqa: E402
    from allength import exact_width as EW  # noqa: E402
CD.KEEP_ZERO_MULTIPLIERS = STRICT

G = {b: EW.chunk_gate_count(None, b, b) for b in range(1, 1025)}
name = "gate_count_table_b_le_1024" + ("_strict" if STRICT else "") + ".pkl"
out = os.path.join(ROOT, "data", name)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(G, f)
print("saved", len(G), "entries to", out, flush=True)
