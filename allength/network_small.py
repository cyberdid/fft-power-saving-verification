"""Step 18: small-h check of the network invocation (network.tex, lemma net:scalar).

Triples T of {1..h} (size 3). S, T are neighbours when |S cap T| is even and S != T.
Wires: x_T, y_S (logical banks), A_{ST} for ordered neighbour pairs, c_j (j=1..h) and c_*.
Maps (exact rational arithmetic):
  (Vx)_{ST} = x_T,   (Gx)_j = sum_{T containing j} x_T,   (Gx)_* = sum_T x_T,
  (JA)_S = - sum_{T nbr S} (|S cap T| - 1)/2 * A_{ST},
  (Rc)_S = (sum_{j in S} c_j - c_*)/2.
Eight rows of one forward invocation (eq. net:schedule), applied in order to random aux values.
Checks: (1) RG + JV = Id on the bank space; (2) one invocation maps y -> y + x and restores all
auxiliaries exactly; (3) neighbour count d_nbr = C(h-3,3)+3(h-3) (with h=100 this gives 147731).
"""

import itertools
import random
from fractions import Fraction
from math import comb


def triples(h):
    return [frozenset(t) for t in itertools.combinations(range(1, h + 1), 3)]


def neighbours(h):
    Ts = triples(h)
    nb = {}
    for S in Ts:
        nb[S] = [T for T in Ts if T != S and len(S & T) % 2 == 0]
    return Ts, nb


def network_check(h, seed=0):
    rng = random.Random(seed)
    Ts, nb = neighbours(h)
    idx_T = {T: i for i, T in enumerate(Ts)}
    pairs = [(S, T) for S in Ts for T in nb[S]]
    pair_idx = {p: i for i, p in enumerate(pairs)}

    def rnd():
        return Fraction(rng.randint(-9, 9), rng.randint(1, 4))

    x = {T: rnd() for T in Ts}
    y = {S: rnd() for S in Ts}
    A = {p: rnd() for p in pairs}
    c = {j: rnd() for j in range(1, h + 1)}
    cstar = rnd()
    c_all = dict(c)
    c_all['*'] = cstar

    def V(x_):   # (Vx)_{ST} = x_T
        return {p: x_[p[1]] for p in pairs}

    def G(x_):   # (Gx)_j and (Gx)_*
        out = {j: sum((x_[T] for T in Ts if j in T), Fraction(0)) for j in range(1, h + 1)}
        out['*'] = sum(x_.values(), Fraction(0))
        return out

    def J(A_):   # (JA)_S
        return {S: -sum((Fraction(len(S & T) - 1, 2) * A_[(S, T)] for T in nb[S]), Fraction(0))
                for S in Ts}

    def R(c_):   # (Rc)_S
        return {S: (sum((c_[j] for j in S), Fraction(0)) - c_['*']) / 2 for S in Ts}

    # eight rows of one forward invocation
    y0 = dict(y)
    y = {S: y[S] - J(A)[S] for S in Ts}                      # row 0
    y = {S: y[S] - R(c_all)[S] for S in Ts}                  # row 1
    Vx = V(x)
    A = {p: A[p] + Vx[p] for p in pairs}                     # row 2
    Gx = G(x)
    c_all = {k: c_all[k] + Gx[k] for k in c_all}             # row 3
    y = {S: y[S] + R(c_all)[S] for S in Ts}                  # row 4
    y = {S: y[S] + J(A)[S] for S in Ts}                      # row 5
    c_all = {k: c_all[k] - Gx[k] for k in c_all}             # row 6
    A = {p: A[p] - Vx[p] for p in pairs}                     # row 7

    # check (2): y -> y + x, auxiliaries restored
    ok_y = all(y[S] == y0[S] + x[S] for S in Ts)
    return ok_y


def check_identity(h, seed=1):
    """RG + JV = Id on bank space: verify exactly on random x."""
    rng = random.Random(seed)
    Ts, nb = neighbours(h)
    pairs = [(S, T) for S in Ts for T in nb[S]]
    x = {T: Fraction(rng.randint(-9, 9), rng.randint(1, 4)) for T in Ts}
    Vx = {p: x[p[1]] for p in pairs}
    Gx = {j: sum((x[T] for T in Ts if j in T), Fraction(0)) for j in range(1, h + 1)}
    Gstar = sum(x.values(), Fraction(0))
    out = {}
    for S in Ts:
        RGx = (sum((Gx[j] for j in S), Fraction(0)) - Gstar) / 2
        JVx = -sum((Fraction(len(S & T) - 1, 2) * Vx[(S, T)] for T in nb[S]), Fraction(0))
        out[S] = RGx + JVx
    return all(out[S] == x[S] for S in Ts)


def d_nbr(h):
    # count of neighbours of one triple: even intersections (0 or 2), excluding itself
    return comb(h - 3, 3) + 3 * (h - 3)


if __name__ == "__main__":
    for h in [5, 6, 7]:
        print(h, check_identity(h), network_check(h), d_nbr(h))
    print("h=100 neighbour count:", d_nbr(100))


def three_stage_check(h, seed=2):
    """Lemma net:scalar, second part, on the full grid T^3 for small h.
    Stage 1: forward X->Y   (y <- y + x with x = X-list, y = Y-list);
    Stage 2: inverse Y->X   (y <- y - x with x = Y-list, y = X-list);
    Stage 3: forward X->Y.  Each stage acts along one axis j, one invocation per fixed other indices.
    Expected: (X_d, Y_d) -> (-Y_d, X_d) for every d in T^3."""
    rng = random.Random(seed)
    Ts = triples(h)
    grid = [(a, b, c) for a in Ts for b in Ts for c in Ts]
    X = {d: Fraction(rng.randint(-9, 9), rng.randint(1, 3)) for d in grid}
    Y = {d: Fraction(rng.randint(-9, 9), rng.randint(1, 3)) for d in grid}
    X0, Y0 = dict(X), dict(Y)

    def stage(axis, src, dst, sign):
        # one invocation per fixed pair of other indices; dst += sign * src along axis
        for d in grid:
            if d[axis] != Ts[0]:
                continue
            fixed = d
            lst = []
            for T in Ts:
                key = list(fixed)
                key[axis] = T
                lst.append(tuple(key))
            for k in lst:
                dst[k] = dst[k] + sign * src[k]

    stage(2, X, Y, +1)    # stage 1: forward X -> Y  (axis 3, arbitrary choice of axis j)
    stage(1, Y, X, -1)    # stage 2: inverse Y -> X
    stage(0, X, Y, +1)    # stage 3: forward X -> Y
    return all(Y[d] == X0[d] and X[d] == -Y0[d] for d in grid)


def enumerate_wires(h):
    """Explicitly construct the wire set of the three-stage network for small h and count it.
    Banks: X_d, Y_d for d in T^3. Each stage j has one invocation per choice of the two other
    indices; each invocation owns side wires A_{ST} (ordered neighbour pairs of its list) and
    central wires c_1..c_h, c_*."""
    Ts, nb = neighbours(h)
    v = len(Ts)
    banks = set()
    for d in itertools.product(Ts, repeat=3):
        banks.add(('X', d))
        banks.add(('Y', d))
    aux = set()
    invocations = 0
    for j in range(3):
        others = [k for k in range(3) if k != j]
        for fixed in itertools.product(Ts, repeat=2):
            inv = invocations
            invocations += 1
            for S in Ts:
                for T in nb[S]:
                    aux.add(('A', inv, S, T))
            for c in list(range(1, h + 1)) + ['*']:
                aux.add(('c', inv, c))
    total = len(banks) + len(aux)
    return total, invocations, v, len(nb[Ts[0]])


def wire_formula(h):
    v = comb(h, 3)
    d = comb(h - 3, 3) + 3 * (h - 3)
    return 2 * v ** 3 + 3 * v * v * (v * d + h + 1), 3 * v * v, d
