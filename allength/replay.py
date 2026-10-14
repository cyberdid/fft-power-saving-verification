"""Step 3: replay lemma with arbitrary borrowed coordinates (local.tex, lemma loc:replay).

Given a linear DAG M : C^e -> C^a (gates with at most two predecessors, outputs are single
reads of a source or gate, or zero), implement the map on disjoint coordinate sets
    (x, y, z)  |->  (x, y + M x, z)
for ARBITRARY initial contents of z (including values correlated with x and y), using only
shear operations  v[dst] += c * v[src]  applied in layers.

Exact arithmetic: Python integers (no rounding), so identities are checked exactly.

Layering: each sweep is split by gate depth; inside a depth level the shear multigraph is
edge-coloured greedily; each colour class is a set of shears on disjoint coordinate pairs and is
applied as one layer. The test checks that the layered word equals the sequential word.
"""

import random
from collections import defaultdict


class DAG:
    """Gates g_0..g_{g-1}; each gate is a list of (kind, idx, coef) with kind in {'x','z'}:
    'x' refers to source x[idx], 'z' to earlier gate idx. Outputs: list of (kind, idx, coef) or None."""

    def __init__(self, e, a, gates, outputs):
        self.e, self.a = e, a
        self.gates = gates
        self.outputs = outputs
        for k, preds in enumerate(gates):
            assert len(preds) <= 2
            for kind, idx, _ in preds:
                if kind == 'z':
                    assert idx < k, "gates must be topologically ordered"

    @property
    def g(self):
        return len(self.gates)

    def evaluate(self, x):
        """Reference evaluation of M x (exact ints)."""
        gv = []
        def val(kind, idx):
            return x[idx] if kind == 'x' else gv[idx]
        for preds in self.gates:
            gv.append(sum(c * val(k, i) for k, i, c in preds))
        out = []
        for o in self.outputs:
            out.append(0 if o is None else o[2] * val(o[0], o[1]))
        return out

    def matrix(self):
        M = [[0] * self.e for _ in range(self.a)]
        for j in range(self.e):
            unit = [1 if t == j else 0 for t in range(self.e)]
            col = self.evaluate(unit)
            for i in range(self.a):
                M[i][j] = col[i]
        return M

    def fan_out(self):
        counts = defaultdict(int)
        for preds in self.gates:
            for k, i, _ in preds:
                counts[(k, i)] += 1
        for o in self.outputs:
            if o is not None:
                counts[(o[0], o[1])] += 1
        return max(counts.values()) if counts else 0

    def depths(self):
        """Longest-path depth of each gate (sources have depth 0, gate depth >= 1)."""
        d = []
        for preds in self.gates:
            dep = 0
            for k, i, _ in preds:
                if k == 'z':
                    dep = max(dep, d[i])
            d.append(dep + 1)
        return d


# Coordinate layout: x = [0, e), y = [e, e+a), z = [e+a, e+a+g)
def _coords(dag):
    e, a = dag.e, dag.a
    def xc(i):
        return i
    def yc(t):
        return e + t
    def zc(k):
        return e + a + k
    return xc, yc, zc


def build_sweeps(dag, drop_sources):
    """Return (forward_shears, output_shears) as lists of (src, dst, coef) for gate sweep and
    output readout. If drop_sources, source edges (x) are omitted in both."""
    xc, yc, zc = _coords(dag)
    forward = []
    for k, preds in enumerate(dag.gates):
        for kind, idx, c in preds:
            if kind == 'x' and drop_sources:
                continue
            src = xc(idx) if kind == 'x' else zc(idx)
            forward.append((src, zc(k), c))
    outs = []
    for t, o in enumerate(dag.outputs):
        if o is None:
            continue
        kind, idx, c = o
        if kind == 'x' and drop_sources:
            continue
        src = xc(idx) if kind == 'x' else zc(idx)
        outs.append((src, yc(t), c))
    return forward, outs


def apply_shears(v, shears):
    for src, dst, c in shears:
        v[dst] += c * v[src]


def undo_shears(v, shears):
    for src, dst, c in reversed(shears):
        v[dst] -= c * v[src]


def replay_sequential(dag, v):
    """The four-sweep word of the proof, applied sequentially. v = x + y + z as one list."""
    xc, yc, zc = _coords(dag)
    fwd, outs = build_sweeps(dag, drop_sources=False)
    # 1. forward sweep, read outputs into y, undo sweep
    apply_shears(v, fwd)
    apply_shears(v, outs)
    undo_shears(v, fwd)
    # 2. same with source edges omitted; subtract readout of FUz, undo
    fwd2, outs2 = build_sweeps(dag, drop_sources=True)
    apply_shears(v, fwd2)
    neg = [(s, d, -c) for s, d, c in outs2]
    apply_shears(v, neg)
    undo_shears(v, fwd2)
    return v


def colour_layers(shears, depth_of_dst):
    """Greedy edge colouring per depth level -> list of layers (each a list of shears)."""
    by_level = defaultdict(list)
    for s in shears:
        by_level[depth_of_dst(s[1])].append(s)
    layers = []
    for level in sorted(by_level):
        pending = by_level[level]
        remaining = list(pending)
        while remaining:
            used = set()
            layer, rest = [], []
            for s in remaining:
                if s[0] in used or s[1] in used:
                    rest.append(s)
                else:
                    layer.append(s)
                    used.add(s[0]); used.add(s[1])
            layers.append(layer)
            remaining = rest
    return layers


def replay_layered(dag, v):
    """Same word, applied as layers. Returns (v, layer_count, max_layer_width_ok)."""
    xc, yc, zc = _coords(dag)
    depths = dag.depths()
    e, a = dag.e, dag.a

    def depth_of(coord):
        if coord < e + a:
            return 0
        return depths[coord - e - a]

    def layered(shears):
        return colour_layers(shears, lambda dst: depth_of(dst))

    fwd, outs = build_sweeps(dag, drop_sources=False)
    fwd_layers = layered(fwd)
    total = 0

    def run(layers, sign=1):
        for L in layers:
            for src, dst, c in L:
                v[dst] += sign * c * v[src]

    # Readout: sources (x, z) are never written here and targets y_t are distinct per output,
    # so the whole readout is one layer.
    run(fwd_layers); total += len(fwd_layers)
    run([outs]); total += 1
    for L in reversed(fwd_layers):
        for src, dst, c in L:
            v[dst] -= c * v[src]
    total += len(fwd_layers)

    fwd2, outs2 = build_sweeps(dag, drop_sources=True)
    fwd2_layers = layered(fwd2)
    run(fwd2_layers); total += len(fwd2_layers)
    neg = [(s, d, -c) for s, d, c in outs2]
    run([neg]); total += 1
    for L in reversed(fwd2_layers):
        for src, dst, c in L:
            v[dst] -= c * v[src]
    total += len(fwd2_layers)
    return v, total


def random_dag(rng, e, a, g, max_coef=3):
    gates = []
    for k in range(g):
        preds = []
        for _ in range(rng.randint(0, 2)):
            if k == 0 or rng.random() < 0.5:
                kind, idx = 'x', rng.randrange(e)
            else:
                kind, idx = 'z', rng.randrange(k)
            c = rng.choice([c for c in range(-max_coef, max_coef + 1) if c != 0])
            preds.append((kind, idx, c))
        gates.append(preds)
    outputs = []
    for _ in range(a):
        if rng.random() < 0.15 or (e + g) == 0:
            outputs.append(None)
            continue
        if g > 0 and rng.random() < 0.5:
            kind, idx = 'z', rng.randrange(g)
        else:
            kind, idx = 'x', rng.randrange(e)
        c = rng.choice([c for c in range(-max_coef, max_coef + 1) if c != 0])
        outputs.append((kind, idx, c))
    return DAG(e, a, gates, outputs)
