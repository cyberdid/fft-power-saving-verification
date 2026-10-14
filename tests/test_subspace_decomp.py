import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import subspace_decomp as SD  # noqa: E402


class ResidualSubspaceDecomposition(unittest.TestCase):
    def test_identity_for_all_orthogonal_odd_pairs(self):
        pairs, holds = SD.check()
        self.assertGreater(pairs, 0)
        self.assertEqual(holds, pairs)


if __name__ == "__main__":
    unittest.main()
