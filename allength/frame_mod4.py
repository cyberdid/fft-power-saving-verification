"""Step 23: mod-4 weight identity in the exact-interface proof (network.tex, proposition net:finite-interface).

For binary u, x in F_2^m with wt(u) odd (so u.u = 1):
    wt(x) - 2*wt(u)*[u.x]  ==  wt(x) + 2*[u.x]   (mod 4),
since 2*wt(u) = 2 (mod 4) when wt(u) is odd. Checked by brute force.
"""

import itertools


def dot(u, x):
    return sum(a & b for a, b in zip(u, x)) % 2


def wt(v):
    return sum(v)


def check(m):
    vecs = list(itertools.product((0, 1), repeat=m))
    count = 0
    for u in vecs:
        if wt(u) % 2 == 0:
            continue
        assert dot(u, u) == 1
        for x in vecs:
            lhs = (wt(x) - 2 * wt(u) * dot(u, x)) % 4
            rhs = (wt(x) + 2 * dot(u, x)) % 4
            if lhs != rhs:
                return False, u, x
            count += 1
    return True, count, None
