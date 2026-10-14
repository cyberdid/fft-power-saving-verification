import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import packing as PK  # noqa: E402


class SectorPacking(unittest.TestCase):
    def test_random_configurations(self):
        self.assertEqual(PK.check(seed=5, trials=300), 300)

    def test_identity_packing_when_all_blocks_singletons(self):
        radices = [2, 3]
        partitions = [[[0], [1]], [[0], [1], [2]]]
        packed, total = PK.brute_packing(radices, partitions)
        self.assertEqual(total, 6)
        self.assertEqual(sorted(packed.values()), list(range(6)))


if __name__ == "__main__":
    unittest.main()
