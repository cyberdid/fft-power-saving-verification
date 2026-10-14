import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import printer as PR  # noqa: E402


class PrinterSelection(unittest.TestCase):
    def test_split_identity(self):
        for a in range(1, 9):
            for b in range(1, 9):
                self.assertTrue(PR.split_identity_holds(a, b), f"a={a} b={b}")

    def test_selection_rule(self):
        # n = 2**20: B = 20, sqrt(B) ~ 4.47 -> limit 4; last record with m in [2,4] wins
        self.assertEqual(PR.select_radix([2, 3, 5, 4, 7, 9], 2 ** 20), 4)
        # no admissible record -> radix 2
        self.assertEqual(PR.select_radix([5, 7, 11], 2 ** 20), 2)
        # n = 4: B = 2, limit 1 -> nothing admissible -> 2
        self.assertEqual(PR.select_radix([3], 4), 2)

    def test_only_first_B_bits_are_visible(self):
        # n = 512: B = 9, limit 3. Records emitted after bit step 9 must not be selected:
        # the 2 at position 10 is invisible, so the last admissible visible record is 3.
        recs = [2] + [3] * 8 + [2]
        self.assertEqual(len(recs), 10)
        self.assertEqual(PR.select_radix(recs, 512), 3)
        # sanity: a 2 at position 8 (inside the first B=9 bit steps) IS selected
        self.assertEqual(PR.select_radix([2] + [3] * 7 + [2], 512), 2)


if __name__ == "__main__":
    unittest.main()
