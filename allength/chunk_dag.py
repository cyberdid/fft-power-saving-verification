"""Step 7: chunk maps as convolution DAGs (local.tex, eqs. loc:two-convolutions, loc:chunk-bounds).

For an outer product (v, w), v of length a and w of length e, the operator
    x (length e) -> c (length a),   b_l = sum_{j>=l} w_{j-l} x_j,   c_i = sum_l v_{i-l} b_l
is built from two convolution DAGs (conv_dag):
  * b_l = conv(x_rev, w)[e-1-l], with x_rev_t = x_{e-1-t} (reversal is relabelling only);
  * c_i = (conv(b, v))[i] for i < a.
A chunk map is a sum of such operators (the three outer products of chunk_displacement).

compose(d1, d2): d2 applied after d1.  sum_dags: parallel DAGs on the same sources, outputs added.
"""

from allength.conv_dag import conv_dag
from allength.replay import DAG


def compose(d1, d2):
    """DAG x -> d2(d1(x)); d2.e must equal len(d1.outputs)."""
    assert d2.e == len(d1.outputs)
    gates = [list(p) for p in d1.gates]
    off = len(gates)

    def through_d1(idx, coef):
        o = d1.outputs[idx]
        if o is None:
            return None
        k1, i1, c1 = o
        return (k1, i1, coef * c1)

    for preds in d2.gates:
        new = []
        for kind, idx, c in preds:
            if kind == 'x':
                r = through_d1(idx, c)
                if r is not None:
                    new.append(r)
            else:
                new.append(('z', off + idx, c))
        gates.append(new)
    outputs = []
    for o in d2.outputs:
        if o is None:
            outputs.append(None)
            continue
        kind, idx, c = o
        if kind == 'x':
            outputs.append(through_d1(idx, c))
        else:
            outputs.append(('z', off + idx, c))
    return DAG(d1.e, d2.a, gates, outputs)


def sum_dags(dags):
    """Parallel sum of DAGs with identical inputs and output sizes."""
    e, a = dags[0].e, dags[0].a
    gates = []
    offsets = []
    for d in dags:
        off = len(gates)
        offsets.append(off)
        for preds in d.gates:
            gates.append([(k, (i if k == 'x' else i + off), c) for k, i, c in preds])
    outputs = []
    for t in range(a):
        terms = []
        for d, off in zip(dags, offsets):
            o = d.outputs[t]
            if o is None:
                continue
            k, i, c = o
            terms.append((k, (i if k == 'x' else i + off), c))
        if not terms:
            outputs.append(None)
            continue
        acc = terms[0]
        for tm in terms[1:]:
            gates.append([(acc[0], acc[1], acc[2]), (tm[0], tm[1], tm[2])])
            acc = ('z', len(gates) - 1, 1)
        outputs.append(acc)
    return DAG(e, a, gates, outputs)


def outer_dag(v, w, e):
    """DAG x (length e) -> c (length len(v)) for one outer product (v, w)."""
    a = len(v)
    # step 1: b = conv(x_rev, w)[e-1-l]; conv_dag input z_t corresponds to x_{e-1-t}
    d1 = conv_dag(w, e)

    def relabel(kind, idx):
        return (e - 1 - idx) if kind == 'x' else idx

    gates1 = [[(k, relabel(k, i), c) for k, i, c in preds] for preds in d1.gates]
    sel = []
    for l in range(e):
        o = d1.outputs[e - 1 - l]
        sel.append(None if o is None else (o[0], relabel(o[0], o[1]), o[2]))
    d1b = DAG(e, e, gates1, sel)
    # step 2: c = first a entries of conv(b, v)
    d2_full = conv_dag(v, e)
    d2 = DAG(e, a, d2_full.gates, d2_full.outputs[:a])
    return compose(d1b, d2)


def chunk_dag(vs, ws, e):
    """Sum of outer-product operators (v_k, w_k), all with inputs of length e."""
    return sum_dags([outer_dag(v, w, e) for v, w in zip(vs, ws)])
