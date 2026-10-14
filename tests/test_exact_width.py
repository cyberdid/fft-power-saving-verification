import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import exact_width as EW  # noqa: E402
from allength.toeplitz_disp import toeplitz_matrix  # noqa: E402


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


class ExactWidthBlock(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(41)

    def rand_h(self, n):
        h = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        h[0] = abs(h[0]) + 1.0
        return h

    def test_block_apply_equals_toeplitz(self):
        for r in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
            h = self.rand_h(r)
            x = self.rng.standard_normal(r) + 1j * self.rng.standard_normal(r)
            b = EW.max_chunk_for_width(r)
            if b < 1:
                # no feasible dense chunk for this width; skip (documented limitation)
                continue
            got = EW.block_toeplitz_apply(h, x, chunk=b, check=True)
            want = toeplitz_matrix(h, r) @ x
            self.assertLess(rel(got, want), 1e-8, f"r={r} chunk={b}")

    def test_smaller_chunks_also_exact(self):
        for r in [8, 16]:
            h = self.rand_h(r)
            x = self.rng.standard_normal(r) + 1j * self.rng.standard_normal(r)
            want = toeplitz_matrix(h, r) @ x
            for b in [1, 2]:
                if b > EW.max_chunk_for_width(r):
                    continue  # dense DAG does not fit the block for this chunk size
                got = EW.block_toeplitz_apply(h, x, chunk=b, check=True)
                self.assertLess(rel(got, want), 1e-8, f"r={r} chunk={b}")

    def test_borrowed_coordinates_restored(self):
        # check=True asserts that borrowed coordinates equal their original values per chunk
        r = 16
        h = self.rand_h(r)
        x = self.rng.standard_normal(r) + 1j * self.rng.standard_normal(r)
        EW.block_toeplitz_apply(h, x, chunk=EW.max_chunk_for_width(r), check=True)

    def test_chunk_feasibility_table(self):
        rows = []
        for r in [8, 16, 32, 64, 128, 256]:
            rows.append((r, EW.max_chunk_for_width(r)))
        print("\nwidth r -> largest feasible chunk b (dense DAG, 2b+g <= r):", rows)
        self.assertGreaterEqual(EW.max_chunk_for_width(64), 4)


if __name__ == "__main__":
    unittest.main()
