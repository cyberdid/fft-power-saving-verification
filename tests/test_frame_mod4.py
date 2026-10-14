import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import frame_mod4 as FM  # noqa: E402


class Mod4WeightIdentity(unittest.TestCase):
    def test_identity_for_odd_weight_u(self):
        for m in range(1, 7):
            ok, info, _ = FM.check(m)
            self.assertTrue(ok, f"m={m}")
            self.assertGreater(info, 0)


if __name__ == "__main__":
    unittest.main()
