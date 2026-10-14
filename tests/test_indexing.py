import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import indexing as IX  # noqa: E402


class IndexingDataMovement(unittest.TestCase):
    def test_table_equals_direct(self):
        self.assertEqual(IX.check(seed=3, trials=300), 300)


if __name__ == "__main__":
    unittest.main()
