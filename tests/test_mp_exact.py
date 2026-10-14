import os
import random
import sys
import unittest

import mpmath as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import mp_exact as M  # noqa: E402


class ExactWidthHighPrecision(unittest.TestCase):
    """Algebraic check of the exact-width block apply in 60-digit arithmetic (fast cases).
    Larger widths (r=128 at 100 digits, r=256 at 200 digits) were run manually; see STATUS."""

    def run_case(self, r, chunk, seed):
        mp.mp.dps = 60
        rnd = random.Random(seed)
        rc = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        h = [rc() + (1 if t == 0 else 0) for t in range(r + 1)]
        x = [rc() for _ in range(r)]
        got = M.mp_block_apply(h, x, chunk=chunk)
        want = [sum(h[i - j] * x[j] for j in range(i + 1)) for i in range(r)]
        num = mp.sqrt(sum(abs(got[i] - want[i]) ** 2 for i in range(r)))
        den = mp.sqrt(sum(abs(want[i]) ** 2 for i in range(r)))
        return num / den

    def test_r8_chunk1(self):
        self.assertLess(self.run_case(8, 1, 1), mp.mpf(10) ** -50)

    def test_r16_chunk1(self):
        self.assertLess(self.run_case(16, 1, 2), mp.mpf(10) ** -50)

    def test_r32_chunk1(self):
        self.assertLess(self.run_case(32, 1, 3), mp.mpf(10) ** -50)


class BlockLocalHighPrecision(unittest.TestCase):
    """Block-local word (admissible chunk per block, borrowing inside the block) in 60 digits."""

    def run_case(self, r, seed):
        mp.mp.dps = 60
        rnd = random.Random(seed)
        rc = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        h = [rc() + (1 if t == 0 else 0) for t in range(r + 1)]
        x = [rc() for _ in range(r)]
        got = M.mp_block_apply_local(h, x)
        want = [sum(h[i - j] * x[j] for j in range(i + 1)) for i in range(r)]
        num = mp.sqrt(sum(abs(got[i] - want[i]) ** 2 for i in range(r)))
        den = mp.sqrt(sum(abs(want[i]) ** 2 for i in range(r)))
        return num / den

    def test_r8(self):
        self.assertLess(self.run_case(8, 21), mp.mpf(10) ** -50)

    def test_r16(self):
        self.assertLess(self.run_case(16, 22), mp.mpf(10) ** -50)

    def test_r32(self):
        self.assertLess(self.run_case(32, 23), mp.mpf(10) ** -50)


if __name__ == "__main__":
    unittest.main()
