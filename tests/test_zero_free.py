import cmath
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import compiler as CP  # noqa: E402


class ZeroFreeSplit(unittest.TestCase):
    def test_split_nonzero_and_sums(self):
        for mu in [0, 1e-17, -0.3 + 2j, 5, 1e6j, 0.7]:
            k, t2 = CP.split_shear(mu)
            self.assertNotEqual(k, 0)
            self.assertNotEqual(t2, 0)
            self.assertLess(abs(k + t2 - mu), 1e-9 * max(1, abs(mu)))

    def test_two_shears_compose_to_target(self):
        def L(c):
            return np.array([[1, 0], [c, 1]], dtype=complex)

        def factor_mat(t):
            M = np.eye(2, dtype=complex)
            for k, (p, q) in enumerate(CP.shear_factors(t)):
                M = M @ np.diag([p, q])
                if k < 3:
                    M = M @ CP.C
            return M

        for mu in [0.7, -0.3 + 1j, 2.0, 1e-17]:
            k, t2 = CP.split_shear(mu)
            self.assertLess(np.max(np.abs(factor_mat(k) @ factor_mat(t2) - L(mu))), 1e-9 * max(1, abs(mu)))


class ZeroFreeCompiledWord(unittest.TestCase):
    def test_equals_fourier_matrix_no_threshold(self):
        rng = np.random.default_rng(81)
        for r in [2, 3, 4, 6, 8]:
            u = cmath.exp(-2j * cmath.pi / r)
            blocks = CP.zero_free_blocks(u, r)
            x = rng.standard_normal(r) + 1j * rng.standard_normal(r)
            err = np.linalg.norm(CP.apply_blocks(blocks, x) - CP.fourier_matrix(u, r) @ x)
            self.assertLess(err / np.linalg.norm(x), 1e-10, f"r={r}")


class EveryPrimitiveRoot(unittest.TestCase):
    def test_all_primitive_roots_up_to_8(self):
        import math
        rng = np.random.default_rng(82)
        for r in range(2, 9):
            for a in range(1, r):
                if math.gcd(a, r) != 1:
                    continue
                u = cmath.exp(2j * cmath.pi * a / r)
                blocks = CP.zero_free_blocks(u, r)
                self.assertTrue(CP.pattern_ok(blocks))
                x = rng.standard_normal(r) + 1j * rng.standard_normal(r)
                err = np.linalg.norm(CP.apply_blocks(blocks, x) - CP.fourier_matrix(u, r) @ x)
                self.assertLess(err / np.linalg.norm(x), 1e-10, f"r={r} a={a}")


class StrictAPIDefault(unittest.TestCase):
    def test_topology_does_not_inherit_global_pruning(self):
        from allength import conv_dag as cd
        from allength import compiler as compiler
        import cmath
        import math
        previous = cd.KEEP_ZERO_MULTIPLIERS
        try:
            u = cmath.exp(2j * math.pi / 16)
            signatures = []
            for flag in (False, True):
                cd.KEEP_ZERO_MULTIPLIERS = flag
                blocks = compiler.zero_free_blocks(u, 16)
                signatures.append(tuple(tuple(tuple(pairs) for pairs in b['pairs']) for b in blocks))
                self.assertEqual(cd.KEEP_ZERO_MULTIPLIERS, flag)
            self.assertEqual(signatures[0], signatures[1])
        finally:
            cd.KEEP_ZERO_MULTIPLIERS = previous


if __name__ == "__main__":
    unittest.main()
