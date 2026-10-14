import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from allength import conv_dag as CD  # noqa: E402
from allength.compiler import transpose_word  # noqa: E402
from allength.exact_field import Cyclotomic, PrimeField, cyclotomic_poly  # noqa: E402
from allength.generic_word import g_block_word_local  # noqa: E402


def dft_mismatches(F, r, corrupt=False):
    H = [F.one()]
    for j in range(1, r + 1):
        H.append(H[-1] * (F.one() - F.root(r, j)))
    cc = [F.root(r, k * (k - 1) // 2) * (1 if k % 2 == 0 else -1) for k in range(r)]
    h = [F.one() / H[j] for j in range(r)] + [F.zero()]
    Dh, Dmid = H[:r], [cc[k] / H[k] for k in range(r)]
    word = g_block_word_local(F, h, r)
    if corrupt:
        for L in word[len(word) // 2:]:
            for k, (s, d, c) in enumerate(L):
                if s is not None:
                    L[k] = (s, d, c + 1)
                    break
            else:
                continue
            break
    tword = transpose_word(word)
    bad = 0
    for j in range(r):
        y = [F.one() if t == j else F.zero() for t in range(r)]
        y = [Dh[i] * y[i] for i in range(r)]
        for L in tword:
            for s, d, c in L:
                if s is None:
                    y[d] = c * y[d]
                else:
                    y[d] = y[d] + c * y[s]
        y = [Dmid[i] * y[i] for i in range(r)]
        for L in word:
            for s, d, c in L:
                if s is None:
                    y[d] = c * y[d]
                else:
                    y[d] = y[d] + c * y[s]
        y = [Dh[i] * y[i] for i in range(r)]
        bad += sum(1 for i in range(r) if not (y[i] == F.root(r, i * j)))
    return bad


class ExactFieldTest(unittest.TestCase):
    def setUp(self):
        self.old = CD.KEEP_ZERO_MULTIPLIERS
        CD.KEEP_ZERO_MULTIPLIERS = True

    def tearDown(self):
        CD.KEEP_ZERO_MULTIPLIERS = self.old

    def test_cyclotomic_polynomials(self):
        self.assertEqual(cyclotomic_poly(8), [1, 0, 0, 0, 1])
        self.assertEqual(cyclotomic_poly(12), [1, 0, -1, 0, 1])

    def test_cyclotomic_inverse_and_roots(self):
        F = Cyclotomic(24)
        z = F.root(24, 1)
        a = F.one() + z * 3 - z * z
        self.assertTrue(a * a.inverse() == F.one())
        p = F.one()
        for _ in range(24):
            p = p * z
        self.assertTrue(p == F.one())
        self.assertFalse(F.root(24, 12) == F.one())

    def test_prime_field_root_order(self):
        F = PrimeField(64)
        self.assertEqual((F.p - 1) % 64, 0)
        w = F.root(64, 1)
        self.assertFalse(F.root(64, 32) == F.one())
        self.assertTrue(F.root(64, 64) == F.one())
        self.assertTrue(w * F.root(64, -1) == F.one())

    def test_exact_dft_small(self):
        self.assertEqual(dft_mismatches(Cyclotomic(64), 8), 0)
        for r in (7, 9, 16):
            n = r * 128 // __import__("math").gcd(r, 128)
            self.assertEqual(dft_mismatches(PrimeField(n), r), 0)

    def test_corruption_is_detected(self):
        self.assertGreater(dft_mismatches(PrimeField(128), 16, corrupt=True), 0)


if __name__ == "__main__":
    unittest.main()
