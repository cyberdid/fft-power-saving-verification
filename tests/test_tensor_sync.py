import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import tensor_sync as TS  # noqa: E402


class SynchronisedTensorFourier(unittest.TestCase):
    def test_two_and_three_axes(self):
        for radices in [(2, 3), (3, 4), (2, 2, 3), (4, 3, 2)]:
            got = TS.tensor_product_result(radices)
            want = TS.tensor_fourier(radices)
            err = np.linalg.norm(got - want) / np.linalg.norm(want)
            self.assertLess(err, 1e-9, f"radices={radices} err={err:.2e}")

    def test_common_length_and_pattern(self):
        words, per_axis = TS.tensor_word((2, 3, 4))
        self.assertEqual(len(words), max(len(m) for m in per_axis))
        for W, r in zip(words, [2, 3, 4]):
            self.assertEqual(W.shape[0], int(np.prod([2, 3, 4])))


if __name__ == "__main__":
    unittest.main()
