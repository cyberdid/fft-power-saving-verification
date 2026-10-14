import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import assembly as AS  # noqa: E402


class AllLengthAssembly(unittest.TestCase):
    def test_all_lengths_2_to_120(self):
        rng = np.random.default_rng(71)
        worst = 0.0
        for n in range(2, 121):
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            e = np.linalg.norm(AS.assembled_dft(x) - np.fft.fft(x)) / np.linalg.norm(np.fft.fft(x))
            worst = max(worst, e)
            self.assertLess(e, 1e-6, f"n={n}")
        print(f"\nworst rel err n=2..120 with compiled kernels: {worst:.2e}")

    def test_selected_composites_and_primes(self):
        rng = np.random.default_rng(72)
        for n in [97, 120, 210, 360, 997, 1000]:
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            e = np.linalg.norm(AS.assembled_dft(x) - np.fft.fft(x)) / np.linalg.norm(np.fft.fft(x))
            self.assertLess(e, 1e-6, f"n={n}")


if __name__ == "__main__":
    unittest.main()
