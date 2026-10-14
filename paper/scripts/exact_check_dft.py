"""Exact (rounding-free) check of the compiled DFT word (article v6, table 1b).

Usage:  python exact_check_dft.py R FIELD [NPRIMES] [NVEC] [--cslots]
  FIELD = cyclo : Q(zeta_N) with exact rational coefficients (a proof by computation of
                  y = F_r(u) x for every basis vector x, i.e. of the whole implemented matrix);
  FIELD = prime : F_p for NPRIMES primes p = 1 mod N (exact, but only a necessary condition).
  N = lcm(R, 2^ceil(log2(8R))) so that u = zeta_R and all power-of-two roots of the
  convolution DAGs (orders < 8R, local.tex) lie in the field.
  Vectors: all R basis vectors, unless NVEC > 0 is given (then NVEC random vectors, F_p only).
--cslots : apply every shear through the zero-free C-slot pattern (split mu = kappa + (mu - kappa),
           kappa = 1 + mu * conj(mu)), with conj(mu) prepared by evaluating the whole word a
           second time with inverse roots (u -> u^-1, every root of order L -> its inverse).
Output: number of output entries checked and number of mismatches (must be 0).
"""
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from allength import conv_dag as CD  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True

from allength.compiler import transpose_word  # noqa: E402
from allength.exact_field import Cyclotomic, PrimeField, lcm  # noqa: E402
from allength.generic_word import g_block_word_local  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
CSLOTS = "--cslots" in sys.argv
R, KIND = int(args[0]), args[1]
NPRIMES = int(args[2]) if len(args) > 2 else 1
NVEC = int(args[3]) if len(args) > 3 else 0
N = lcm(R, 2 ** math.ceil(math.log2(8 * R)))


class Inverted:
    """The same field with every root replaced by its inverse (image of complex conjugation)."""

    def __init__(self, F):
        self.F = F

    def one(self):
        return self.F.one()

    def zero(self):
        return self.F.zero()

    def root(self, L, k):
        return self.F.root(L, -k)


def fourier(F):
    u = F.root(R, 1)
    H = [F.one()]
    for j in range(1, R + 1):
        H.append(H[-1] * (F.one() - F.root(R, j)))
    cc = [F.root(R, k * (k - 1) // 2) * (1 if k % 2 == 0 else -1) for k in range(R)]
    # H_R = 0 because 1 - u^R = 0; the width-R Toeplitz matrix uses only h_0..h_{R-1}.
    # h_R is set to 0: if the word read it, the exact comparison below would fail.
    h = [F.one() / H[j] for j in range(R)] + [F.zero()]
    return u, H[:R], [cc[k] / H[k] for k in range(R)], h


def shear_slots(F, y, s, d, t):
    i = F.root(4, 1)
    a_c, b_c = (F.one() + i) / 2, (F.one() - i) / 2
    D = [(-(5 * i) / (3 * t), F.one()), (F.from_int(2), F.one()), (F.from_int(3), -F.one()),
         ((2 * t) / (5 * (F.one() + i)), F.one() / (2 * (F.one() - i)))]
    a, b = y[s], y[d]
    for k in (3, 2, 1, 0):
        p, q = D[k]
        a, b = p * a, q * b
        if k > 0:
            a, b = a_c * a + b_c * b, b_c * a + a_c * b
    y[s], y[d] = a, b


def apply_word(F, y, word, word_bar):
    for li, L in enumerate(word):
        Lb = word_bar[li] if word_bar is not None else None
        for gi, (s, d, c) in enumerate(L):
            if s is None:
                y[d] = c * y[d]
            elif word_bar is None:
                y[d] = y[d] + c * y[s]
            else:
                kappa = F.one() + c * Lb[gi][2]
                shear_slots(F, y, s, d, kappa)
                shear_slots(F, y, s, d, c - kappa)
    return y


def check(F, label):
    t0 = time.time()
    u, Dh, Dmid, h = fourier(F)
    word = g_block_word_local(F, h, R)
    tword = transpose_word(word)
    word_b = tword_b = None
    if CSLOTS:
        Fi = Inverted(F)
        _, _, _, hb = fourier(Fi)
        word_b = g_block_word_local(Fi, hb, R)
        same = len(word_b) == len(word) and all(
            [(s, d) for s, d, _ in L] == [(s, d) for s, d, _ in Lb] for L, Lb in zip(word, word_b))
        assert same, "conjugate word has a different structure"
        tword_b = transpose_word(word_b)
    t_build = time.time() - t0
    rnd = random.Random(R)
    if NVEC > 0:
        assert isinstance(F, PrimeField)
        xs = [[F.from_int(rnd.randrange(F.p)) for _ in range(R)] for _ in range(NVEC)]
        kind = f"random({NVEC})"
    else:
        xs = [[F.one() if t == j else F.zero() for t in range(R)] for j in range(R)]
        kind = f"basis({R})"
    upow = [F.root(R, k) for k in range(R)]
    checked = mismatches = 0
    for x in xs:
        y = [Dh[i] * x[i] for i in range(R)]
        y = apply_word(F, y, tword, tword_b)
        y = [Dmid[i] * y[i] for i in range(R)]
        y = apply_word(F, y, word, word_b)
        y = [Dh[i] * y[i] for i in range(R)]
        for i in range(R):
            want = F.zero()
            for j in range(R):
                want = want + upow[(i * j) % R] * x[j]
            checked += 1
            if not (y[i] == want):
                mismatches += 1
    print(f"exact r={R} {label} {'C-slots+conj-DAG' if CSLOTS else 'word'} {kind}: "
          f"entries checked={checked} mismatches={mismatches} layers={len(word)} "
          f"build={t_build:.0f}s total={time.time() - t0:.0f}s", flush=True)
    return mismatches


bad = 0
if KIND == "cyclo":
    F = Cyclotomic(N)
    bad += check(F, f"Q(zeta_{N}) [degree {F.d}]")
else:
    for idx in range(NPRIMES):
        F = PrimeField(N, index=idx)
        bad += check(F, f"F_p p={F.p} (p = 1 mod {N})")
sys.exit(1 if bad else 0)
