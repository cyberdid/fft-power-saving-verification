"""Step 26: subspace decomposition used in the residual list (network.tex, lemma net:residuals).

Over F_2^n with t_X, t_Y orthogonal of odd weight (t.t = 1, t_X.t_Y = 0):
    t_Y^perp = <t_X>  (+)  <t_X, t_Y>^perp .
"""

import itertools


def dot(a, b):
    return sum(x & y for x, y in zip(a, b)) % 2


def span(vs, n):
    S = {tuple([0] * n)}
    for v in vs:
        S |= {tuple((x + y) % 2 for x, y in zip(s, v)) for s in S}
    return S


def perp(S, n):
    return {x for x in itertools.product((0, 1), repeat=n) if all(dot(x, s) == 0 for s in S)}


def check(n_range=range(3, 7)):
    pairs = holds = 0
    for n in n_range:
        vecs = list(itertools.product((0, 1), repeat=n))
        for tX in vecs:
            for tY in vecs:
                if dot(tX, tX) != 1 or dot(tY, tY) != 1 or dot(tX, tY) != 0 or tX == tY:
                    continue
                pairs += 1
                lhs = perp([tY], n)
                S1 = span([tX], n)
                S2 = perp([tX, tY], n)
                rhs = {tuple((a + b) % 2 for a, b in zip(s, t)) for s in S1 for t in S2}
                holds += (lhs == rhs)
    return pairs, holds
