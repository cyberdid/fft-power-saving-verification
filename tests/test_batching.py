import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import batching as BA  # noqa: E402


class CriticalExponent(unittest.TestCase):
    def test_recurrence_grows_like_k_to_theta(self):
        for m, lam in [(4, 3.5), (6, 5.2), (10, 9.9)]:
            theta, lo, hi = BA.growth_ratio(m, lam)
            self.assertGreater(lo, 0)
            # ratio stays within a bounded band: no extra log factor
            self.assertLess(hi / lo, 20.0, f"m={m} lam={lam}")

    def test_theta_is_in_unit_interval(self):
        import math
        for m, lam in [(4, 3.5), (6, 5.2), (10, 9.9)]:
            theta = math.log(lam) / math.log(m)
            self.assertTrue(0 < theta < 1)


if __name__ == "__main__":
    unittest.main()
