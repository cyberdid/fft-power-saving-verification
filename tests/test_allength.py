import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import dft_all as D  # noqa: E402


def rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


class AllLengthDFT(unittest.TestCase):
    def test_all_lengths_1_to_200(self):
        rng = np.random.default_rng(1)
        worst = 0.0
        for n in range(1, 201):
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            e = rel_err(D.dft_all(x), np.fft.fft(x))
            worst = max(worst, e)
            self.assertLess(e, 1e-10, f"n={n}")
        print(f"\nworst rel err n=1..200: {worst:.3e}")

    def test_selected_lengths(self):
        rng = np.random.default_rng(2)
        # prime, prime power, composite with coprime split, large composite, power of two
        for n in [97, 127, 256, 360, 997, 1000, 1024, 1536, 2310, 4096]:
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            e = rel_err(D.dft_all(x), np.fft.fft(x))
            print(f"n={n:5d} rel err {e:.3e}")
            self.assertLess(e, 1e-9, f"n={n}")

    def test_good_thomas_identity_small(self):
        rng = np.random.default_rng(3)
        for n1, n2 in [(3, 4), (5, 7), (8, 9), (15, 16)]:
            n = n1 * n2
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            self.assertLess(rel_err(D.good_thomas(x, n1, n2), np.fft.fft(x)), 1e-11, f"{n1}x{n2}")

    def test_bluestein_identity(self):
        rng = np.random.default_rng(4)
        for n in [7, 33, 97, 101, 250]:
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            self.assertLess(rel_err(D.bluestein(x), np.fft.fft(x)), 1e-10, f"n={n}")

    def test_impulse_and_constant(self):
        for n in [6, 12, 100, 1000]:
            imp = np.zeros(n, dtype=complex)
            imp[0] = 1
            self.assertLess(rel_err(D.dft_all(imp), np.ones(n, dtype=complex)), 1e-12)
            ones = np.ones(n, dtype=complex)
            want = np.zeros(n, dtype=complex)
            want[0] = n
            self.assertLess(rel_err(D.dft_all(ones), want), 1e-12)


if __name__ == "__main__":
    unittest.main()
