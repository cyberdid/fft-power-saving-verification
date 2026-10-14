import os
import sys
import unittest

import mpmath as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import newton_local as NL  # noqa: E402

TOL = mp.mpf(10) ** (-60)


class NewtonReductionExact(unittest.TestCase):
    def setUp(self):
        # Isolate precision from global mp.mp.dps left by other test modules under discover.
        self._dps = mp.mp.dps
        mp.mp.dps = 80

    def tearDown(self):
        mp.mp.dps = self._dps

    def test_factorisation_identity_all_small_r(self):
        worst = mp.mpf(0)
        for r in range(1, 25):
            u = NL.primitive_root(r, 1)
            res = NL.factorisation_residual(r, u)
            worst = max(worst, res)
            self.assertLess(res, TOL, f"r={r}")
        print(f"\nexact identity worst residual r=1..24: {mp.nstr(worst, 3)}")

    def test_factorisation_other_primitive_roots(self):
        import math
        for r in [7, 9, 12, 15, 20]:
            for a in range(1, r):
                if math.gcd(a, r) != 1:
                    continue
                u = NL.primitive_root(r, a)
                self.assertLess(NL.factorisation_residual(r, u), TOL, f"r={r}, a={a}")

    def test_closed_form_matches_product(self):
        for r in [1, 2, 3, 5, 8, 13, 16]:
            u = NL.primitive_root(r, 1)
            self.assertLess(NL.closed_form_residual(r, u), TOL, f"r={r}")

    def test_toeplitz_structure(self):
        for r in [4, 9, 16]:
            u = NL.primitive_root(r, 1)
            self.assertTrue(NL.toeplitz_entries_ok(r, u))

    def test_lower_triangular(self):
        r = 12
        u = NL.primitive_root(r, 1)
        N = NL.newton_N(r, u)
        for i in range(r):
            for k in range(i + 1, r):
                self.assertLess(abs(N[i, k]), TOL)

    def test_larger_r_exact(self):
        for r in [32, 48, 64]:
            u = NL.primitive_root(r, 1)
            self.assertLess(NL.factorisation_residual(r, u), mp.mpf(10) ** (-40), f"r={r}")


if __name__ == "__main__":
    unittest.main()
