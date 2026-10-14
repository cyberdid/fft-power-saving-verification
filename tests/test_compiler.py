import cmath
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import compiler as CP  # noqa: E402


class CompiledFourier(unittest.TestCase):
    def test_shear_factorisation_residual(self):
        for c in [0.7, -1.3 + 0.4j, 2.5, 0.1j, 1 - 3j, 1e-6, 1e6]:
            self.assertLess(CP.shear_factor_residual(c), 1e-10 * max(1, abs(c)), f"c={c}")

    def test_compiled_word_equals_fourier_matrix(self):
        rng = np.random.default_rng(51)
        for r in [2, 3, 4, 5, 6, 8]:
            u = cmath.exp(2j * cmath.pi / r)
            slots = CP.compile_fourier(u, r)
            x = rng.standard_normal(r) + 1j * rng.standard_normal(r)
            got = CP.apply_slots(slots, x)
            want = CP.fourier_matrix(u, r) @ x
            err = np.linalg.norm(got - want) / np.linalg.norm(want)
            self.assertLess(err, 1e-9, f"r={r}")

    def test_slot_types(self):
        r = 6
        u = cmath.exp(2j * cmath.pi / r)
        for kind, data in CP.compile_fourier(u, r):
            if kind == 'diag':
                self.assertEqual(len(data), r)
            else:
                self.assertEqual(kind, 'C')
                coords = [c for pair in data for c in pair]
                self.assertEqual(len(coords), len(set(coords)), "C slot pairs must be disjoint")

    def test_dropped_C_slot_is_detected(self):
        # mutation check: removing one C slot must break the identity
        r = 4
        u = cmath.exp(2j * cmath.pi / r)
        slots = CP.compile_fourier(u, r)
        idx = next(k for k, (kind, _) in enumerate(slots) if kind == 'C')
        mutated = slots[:idx] + slots[idx + 1:]
        x = np.arange(1, r + 1, dtype=complex)
        err = np.linalg.norm(CP.apply_slots(mutated, x) - CP.fourier_matrix(u, r) @ x)
        self.assertGreater(err, 1e-3)


class FixedPattern(unittest.TestCase):
    def test_blocks_equal_fourier_matrix(self):
        rng = np.random.default_rng(61)
        for r in [2, 4, 6, 8]:
            u = cmath.exp(2j * cmath.pi / r)
            blocks = CP.compile_blocks(u, r)
            self.assertTrue(CP.pattern_ok(blocks))
            x = rng.standard_normal(r) + 1j * rng.standard_normal(r)
            err = np.linalg.norm(CP.apply_blocks(blocks, x) - CP.fourier_matrix(u, r) @ x)
            self.assertLess(err / np.linalg.norm(x), 1e-10, f"r={r}")

    def test_pattern_detects_wrong_block(self):
        r = 4
        blocks = CP.compile_blocks(cmath.exp(2j * cmath.pi / r), r)
        blocks[3]['diag'].pop()          # break the fixed pattern
        self.assertFalse(CP.pattern_ok(blocks))


class ExactlyRepresentedRoots(unittest.TestCase):
    def test_all_float_compilers_accept_minus_one_and_i(self):
        for r,u in ((2,-1+0j),(4,1j)):
            expected=CP.fourier_matrix(u,r)
            for compile_word,apply in ((CP.compile_fourier,CP.apply_slots),(CP.compile_blocks,CP.apply_blocks),(CP.zero_free_blocks,CP.apply_blocks)):
                compiled=compile_word(u,r)
                got=np.column_stack([apply(compiled,np.eye(r,dtype=complex)[:,j]) for j in range(r)])
                self.assertLess(np.max(np.abs(got-expected)),1e-10)


if __name__ == "__main__":
    unittest.main()
