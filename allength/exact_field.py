"""Exact fields for checking the compiled word without rounding (article v6, section 6.1b).

Two fields expose the same small interface (one, zero, root(L, k) = exp(2 pi i k / L) image,
elements with + - * / and ==):

  Cyclotomic(N): Q(zeta_N) with exact rational coefficients modulo the N-th cyclotomic
                 polynomial. Equality here is equality of complex numbers (zeta_N -> e^{2 pi i/N}).
  PrimeField(N, p): F_p with p = 1 mod N and a fixed element omega of order N. The map
                 zeta_N -> omega is a ring homomorphism from Z[zeta_N][1/denominators], so every
                 identity over Q(zeta_N) holds in F_p as well; a check in F_p is exact
                 (no rounding) but is a necessary condition, not a proof of the complex identity.
"""
from fractions import Fraction
from math import gcd


# ----------------------------------------------------------------------------- polynomials
def _trim(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if y:
                out[i + j] += x * y
    return out


def _pdivmod(a, b):
    """Polynomial division over Q (lists low -> high)."""
    a = [Fraction(x) for x in a]
    b = _trim([Fraction(x) for x in b])
    q = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    lead = b[-1]
    while len(_trim(a)) >= len(b) and any(a):
        shift = len(a) - len(b)
        f = a[-1] / lead
        q[shift] = f
        for i, y in enumerate(b):
            a[i + shift] -= f * y
        _trim(a)
        if len(a) < len(b):
            break
    return q, a


def _int_divexact(a, b):
    q, rem = _pdivmod(a, b)
    assert all(x == 0 for x in rem)
    return [int(x) for x in q]


def cyclotomic_poly(N):
    """Integer coefficients of Phi_N (low -> high), by Phi_N = (x^N - 1) / prod_{d|N, d<N} Phi_d."""
    cache = {}

    def phi(n):
        if n in cache:
            return cache[n]
        num = [-1] + [0] * (n - 1) + [1]
        for d in range(1, n):
            if n % d == 0:
                num = _int_divexact(num, phi(d))
        cache[n] = num
        return num
    return phi(N)


# ----------------------------------------------------------------------------- Q(zeta_N)
class Cyclotomic:
    def __init__(self, N):
        self.N = N
        self.phi = cyclotomic_poly(N)
        self.d = len(self.phi) - 1
        # x^k mod Phi_N for k < 2d (reduction table)
        self._red = {}

    def elem(self, coeffs):
        return CycloElem(self, self._reduce(list(coeffs)))

    def _reduce(self, c):
        d, phi = self.d, self.phi
        c = [Fraction(x) for x in c]
        for k in range(len(c) - 1, d - 1, -1):
            f = c[k]
            if f:
                # x^k = x^(k-d) * x^d, x^d = -sum_{i<d} phi_i x^i (Phi monic)
                for i in range(d):
                    if phi[i]:
                        c[k - d + i] -= f * phi[i]
            c[k] = Fraction(0)
        c = c[:d] + [Fraction(0)] * (d - len(c[:d]))
        return tuple(c)

    def one(self):
        return self.elem([1])

    def zero(self):
        return self.elem([0])

    def root(self, L, k):
        assert self.N % L == 0
        e = (k * (self.N // L)) % self.N
        return self.elem([0] * e + [1])

    def from_int(self, n):
        return self.elem([n])


class CycloElem:
    __slots__ = ("F", "c")

    def __init__(self, F, c):
        self.F, self.c = F, c

    def _co(self, o):
        if isinstance(o, CycloElem):
            return o
        return self.F.elem([o])

    def __add__(self, o):
        o = self._co(o)
        return CycloElem(self.F, tuple(a + b for a, b in zip(self.c, o.c)))
    __radd__ = __add__

    def __sub__(self, o):
        o = self._co(o)
        return CycloElem(self.F, tuple(a - b for a, b in zip(self.c, o.c)))

    def __rsub__(self, o):
        return self._co(o) - self

    def __neg__(self):
        return CycloElem(self.F, tuple(-a for a in self.c))

    def __mul__(self, o):
        if not isinstance(o, CycloElem):
            o = Fraction(o)
            return CycloElem(self.F, tuple(a * o for a in self.c))
        return CycloElem(self.F, self.F._reduce(_pmul(list(self.c), list(o.c))))
    __rmul__ = __mul__

    def inverse(self):
        """Extended Euclid in Q[x]: find s with s*a = 1 mod Phi_N (Phi_N irreducible)."""
        a = _trim([Fraction(x) for x in self.c])
        assert any(a), "division by zero in Q(zeta_N)"

        def psub(p, q):
            n = max(len(p), len(q))
            return _trim([(p[i] if i < len(p) else 0) - (q[i] if i < len(q) else 0)
                          for i in range(n)])

        old_r, r = a, _trim([Fraction(x) for x in self.F.phi])
        old_s, s = [Fraction(1)], [Fraction(0)]
        while any(r):
            q, rem = _pdivmod(old_r, r)
            rem = _trim(rem) if rem else [Fraction(0)]
            old_r, r = r, rem
            old_s, s = s, psub(old_s, _pmul(q, s))
        assert len(old_r) == 1 and old_r[0] != 0, "gcd is not a constant"
        inv = self.F.elem([x / old_r[0] for x in old_s])
        assert (self * inv) == self.F.one(), "inverse check failed"
        return inv

    def __truediv__(self, o):
        if not isinstance(o, CycloElem):
            return self * Fraction(1, 1) * (1 / Fraction(o))
        return self * o.inverse()

    def __rtruediv__(self, o):
        return self._co(o) * self.inverse()

    def __eq__(self, o):
        return self.c == self._co(o).c

    def __hash__(self):
        return hash(self.c)

    def is_zero(self):
        return not any(self.c)


# ----------------------------------------------------------------------------- F_p
def _is_prime(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):      # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime_factors(n):
    out, q = set(), 2
    while q * q <= n:
        while n % q == 0:
            out.add(q)
            n //= q
        q += 1
    if n > 1:
        out.add(n)
    return out


class PrimeField:
    def __init__(self, N, start=2 ** 61, index=0):
        """Smallest primes p = 1 mod N above `start`; `index` selects the 0th, 1st, ... one."""
        self.N = N
        k = start // N + 1
        found = -1
        while True:
            p = k * N + 1
            if _is_prime(p):
                found += 1
                if found == index:
                    break
            k += 1
        self.p = p
        qs = _prime_factors(N)
        g = 2
        while True:
            w = pow(g, (p - 1) // N, p)
            if all(pow(w, N // q, p) != 1 for q in qs):
                break
            g += 1
        self.omega = w

    def one(self):
        return GF(self, 1)

    def zero(self):
        return GF(self, 0)

    def root(self, L, k):
        assert self.N % L == 0
        return GF(self, pow(self.omega, (k * (self.N // L)) % self.N, self.p))

    def from_int(self, n):
        return GF(self, n % self.p)


class GF:
    __slots__ = ("F", "v")

    def __init__(self, F, v):
        self.F, self.v = F, v % F.p

    def _co(self, o):
        return o.v if isinstance(o, GF) else o

    def __add__(self, o):
        return GF(self.F, self.v + self._co(o))
    __radd__ = __add__

    def __sub__(self, o):
        return GF(self.F, self.v - self._co(o))

    def __rsub__(self, o):
        return GF(self.F, self._co(o) - self.v)

    def __neg__(self):
        return GF(self.F, -self.v)

    def __mul__(self, o):
        return GF(self.F, self.v * self._co(o))
    __rmul__ = __mul__

    def __truediv__(self, o):
        b = self._co(o) % self.F.p
        assert b != 0, "division by zero in F_p (denominator vanishes mod p)"
        return GF(self.F, self.v * pow(b, -1, self.F.p))

    def __rtruediv__(self, o):
        assert self.v != 0, "division by zero in F_p"
        return GF(self.F, self._co(o) * pow(self.v, -1, self.F.p))

    def __eq__(self, o):
        return self.v == self._co(o) % self.F.p

    def __hash__(self):
        return hash(self.v)

    def is_zero(self):
        return self.v == 0


def lcm(a, b):
    return a * b // gcd(a, b)
