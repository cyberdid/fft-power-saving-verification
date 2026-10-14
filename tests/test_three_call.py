import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import compiler as CP  # noqa: E402


def upper_shear_factors(c):
    """[[1,c],[0,1]] = D3 C D2 C D1 C D0 : transpose of the lower-shear factorisation
    (C and diagonal matrices are symmetric, so transposition reverses the order)."""
    f = CP.shear_factors(c)
    return list(reversed(f))


class ThreeCallWord(unittest.TestCase):
    def test_I_plus_E12_three_calls(self):
        # lemma syn:local-certificate, second clause, with c = 1
        f = upper_shear_factors(1.0)
        M = np.eye(2, dtype=complex)
        for k, (p, q) in enumerate(f):
            M = M @ np.diag([p, q])
            if k < 3:
                M = M @ CP.C
        U = np.array([[1, 1], [0, 1]], dtype=complex)
        self.assertLess(np.max(np.abs(M - U)), 1e-12)

    def test_upper_shear_general_c(self):
        for c in [0.5, -2.0 + 1j, 3j]:
            f = upper_shear_factors(c)
            M = np.eye(2, dtype=complex)
            for k, (p, q) in enumerate(f):
                M = M @ np.diag([p, q])
                if k < 3:
                    M = M @ CP.C
            self.assertLess(np.max(np.abs(M - np.array([[1, c], [0, 1]]))), 1e-10 * max(1, abs(c)))


if __name__ == "__main__":
    unittest.main()
