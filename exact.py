"""Finite paper identities over Q(i), not the full sub-n-log-n construction."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import comb


@dataclass(frozen=True)
class Q:
    r: Fraction = Fraction(0)
    i: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, 'r', Fraction(self.r))
        object.__setattr__(self, 'i', Fraction(self.i))

    def __add__(self, other):
        other = other if isinstance(other, Q) else Q(other)
        return Q(self.r+other.r, self.i+other.i)

    __radd__ = __add__

    def __neg__(self):
        return Q(-self.r, -self.i)

    def __sub__(self, other):
        return self + -(other if isinstance(other, Q) else Q(other))

    def __rsub__(self, other):
        return Q(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Q) else Q(other)
        return Q(self.r*other.r-self.i*other.i, self.r*other.i+self.i*other.r)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Q) else Q(other)
        denominator = other.r**2 + other.i**2
        if not denominator:
            raise ZeroDivisionError('zero Gaussian rational')
        return Q((self.r*other.r+self.i*other.i)/denominator,
                 (self.i*other.r-self.r*other.i)/denominator)

    def __pow__(self, power):
        if power < 0:
            return Q(1) / self**(-power)
        result, base = Q(1), self
        while power:
            if power & 1:
                result = result*base
            base = base*base
            power >>= 1
        return result

    def encoded(self):
        return {'real': str(self.r), 'imag': str(self.i)}


A = Q(Fraction(1, 2), Fraction(1, 2))
B = Q(Fraction(1, 2), Fraction(-1, 2))


def c_pair_naive(x, y):
    return A*x+B*y, B*x+A*y


def c_pair(x, y):
    # Elementary local identity: one scalar multiply, three add/subtracts.
    # This familiar simplification is NOT the paper's network dimension saving.
    t = B*(y-x)
    return x+t, y-t


def directional(values, z, inverse=False):
    n = len(values)
    if n < 2 or n & (n-1) or not 0 < z < n:
        raise ValueError('power-of-two array and nonzero in-range XOR direction required')
    output = list(values)
    for index in range(n):
        peer = index ^ z
        if index < peer:
            u, v = c_pair(values[index], values[peer])
            output[index], output[peer] = (v, u) if inverse else (u, v)
    return output


def tensor_c(values):
    n = len(values)
    if n < 1 or n & (n-1):
        raise ValueError('power-of-two length required')
    output = list(values)
    for bit in range(n.bit_length()-1):
        output = directional(output, 1 << bit)
    return output


def frame_transform(values, basis):
    """Direct H diag(i^wt(P_U x)) H / n, exact; orthonormal GF(2) basis only."""
    n = len(values)
    if not n or n & (n-1):
        raise ValueError('power-of-two frame width required')
    if any(not 0 < z < n or z.bit_count() % 2 != 1 for z in basis):
        raise ValueError('basis vectors must be in-range binary norm-one vectors')
    if any((x & y).bit_count() % 2 for x, y in combinations(basis, 2)):
        raise ValueError('basis must be pairwise orthogonal over GF(2)')
    def walsh(v):
        return [sum(((-1 if (x & y).bit_count() % 2 else 1)*value
                     for y, value in enumerate(v)), Q()) for x in range(n)]
    transformed = walsh(values)
    for x in range(n):
        projection = 0
        for z in basis:
            if (x & z).bit_count() % 2:
                projection ^= z
        transformed[x] = transformed[x]*(Q(0, 1)**(projection.bit_count() % 4))
    return [value/n for value in walsh(transformed)]


class ScalarNetwork:
    """Rows 0–7 of network.tex, reduced h; auxiliary values need not be zero."""
    def __init__(self, h):
        if not isinstance(h, int) or not 3 <= h <= 10:
            raise ValueError('explicit toy network supports 3 <= h <= 10')
        self.h = h
        self.triples = [frozenset(t) for t in combinations(range(h), 3)]
        self.v = len(self.triples)
        self.edges = [(s, t) for s, S in enumerate(self.triples)
                      for t, T in enumerate(self.triples) if len(S & T) % 2 == 0]
        self.coefficients = [Fraction(1-len(self.triples[s] & self.triples[t]), 2)
                             for s, t in self.edges]

    def _J(self, side):
        result = [Q() for _ in self.triples]
        for (s, _), coefficient, value in zip(self.edges, self.coefficients, side):
            result[s] = result[s]+coefficient*value
        return result

    def _R(self, center):
        return [(sum((center[j] for j in S), Q())-center[-1])/2
                for S in self.triples]

    def _G(self, source):
        result = [Q() for _ in range(self.h+1)]
        for T, value in zip(self.triples, source):
            for j in T:
                result[j] = result[j]+value
            result[-1] = result[-1]+value
        return result

    def invoke(self, source, target, side, center, inverse=False):
        if not (len(source) == len(target) == self.v and
                len(side) == len(self.edges) and len(center) == self.h+1):
            raise ValueError('network coordinate dimensions do not match')
        x, y, aux, c = list(source), list(target), list(side), list(center)
        rows = range(7, -1, -1) if inverse else range(8)
        flip = -1 if inverse else 1
        for row in rows:
            if row in (0, 5):
                delta, sign = self._J(aux), (-1 if row == 0 else 1)*flip
                y = [a+sign*b for a, b in zip(y, delta)]
            elif row in (1, 4):
                delta, sign = self._R(c), (-1 if row == 1 else 1)*flip
                y = [a+sign*b for a, b in zip(y, delta)]
            elif row in (2, 7):
                sign = (1 if row == 2 else -1)*flip
                aux = [a+sign*x[t] for a, (_, t) in zip(aux, self.edges)]
            else:
                sign = (1 if row == 3 else -1)*flip
                c = [a+sign*b for a, b in zip(c, self._G(x))]
        return x, y, aux, c


def signed_bank_exchange(h=4):
    """Three stages, each invocation with fresh dirty correlated auxiliaries."""
    net = ScalarNetwork(h)
    keys = list(product(range(net.v), repeat=3))
    X = {d: Q(index-9, index % 5) for index, d in enumerate(keys)}
    Y = {d: Q(2-index, -index % 7) for index, d in enumerate(keys)}
    original_x, original_y = X.copy(), Y.copy()
    invocation_count = 0
    for axis in range(3):
        for fixed in product(range(net.v), repeat=2):
            selected = []
            for variable in range(net.v):
                d, f = [], iter(fixed)
                for j in range(3):
                    d.append(variable if j == axis else next(f))
                selected.append(tuple(d))
            source_bank, target_bank = (Y, X) if axis == 1 else (X, Y)
            x = [source_bank[d] for d in selected]
            y = [target_bank[d] for d in selected]
            side = [x[t]+Q(index, -index) for index, (_, t) in enumerate(net.edges)]
            center = [y[index % net.v]+Q(index, 1) for index in range(h+1)]
            unchanged_x, result, restored_side, restored_center = net.invoke(
                x, y, side, center, inverse=(axis == 1))
            if unchanged_x != x or restored_side != side or restored_center != center:
                raise AssertionError('scalar invocation corrupted borrowed contents')
            for d, value in zip(selected, result):
                target_bank[d] = value
            invocation_count += 1
    if X != {d: -original_y[d] for d in keys} or Y != original_x:
        raise AssertionError('three-stage signed exchange identity failed')
    return {'h': h, 'bank_entries': len(keys), 'invocations': invocation_count,
            'exact_signed_exchange': True, 'all_dirty_auxiliaries_restored': True,
            'scope': 'scalar network only; no binary frame labels or tensor saving'}


def newton_factors(n):
    roots = {1: Q(1), 2: Q(-1), 4: Q(0, -1)}
    if n not in roots:
        raise ValueError('exact Q(i) DFT supports lengths 1, 2, 4 only')
    u = roots[n]
    H = [Q(1)]
    for j in range(1, n):
        H.append(H[-1]*(Q(1)-u**j))
    N = [[Q() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(i+1):
            N[i][k] = ((-1)**k)*(u**(k*(k-1)//2))*H[i]/H[i-k]
    return u, N, [N[i][i] for i in range(n)]


def newton_dft(values):
    n = len(values)
    _, N, diagonal = newton_factors(n)
    # Transpose, NOT conjugate-transpose. Fourier matrix is symmetric.
    middle = [sum((N[j][i]*values[j] for j in range(i, n)), Q())/diagonal[i]
              for i in range(n)]
    return [sum((N[i][j]*middle[j] for j in range(i+1)), Q()) for i in range(n)]


def direct_dft(values):
    u, _, _ = newton_factors(len(values))
    return [sum((u**(i*j)*x for j, x in enumerate(values)), Q())
            for i in range(len(values))]


def network_budget(h):
    if not isinstance(h, int) or h < 3:
        raise ValueError('integer h >= 3 required')
    v, m = comb(h, 3), h**3
    degree = (comb(h-3, 3) if h >= 6 else 0)+3*(h-3)
    n0, i0 = v**3, 3*v**2
    W = 2*n0+i0*(v*degree+h+1)
    delta = 2*n0-2*i0*(h+1)*h
    s = W*m-delta
    W_star = 1 << (W-1).bit_length()
    return {'h': h, 'v': v, 'm': m, 'neighbor_degree': degree, 'N0': n0,
            'I0': i0, 'W': W, 's': s, 'delta': delta, 'W_star': W_star,
            'S': W_star*m-delta, 'K': m*((W_star.bit_length()-1)+1),
            'strict_dimension_margin': delta > 0,
            'scope': 'integer accounting; not a full network proof or execution'}
