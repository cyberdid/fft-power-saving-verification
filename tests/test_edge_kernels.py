import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import edge_kernels as EK  # noqa: E402


class DirectionalKernelIdentities(unittest.TestCase):
    def test_scalar_identities(self):
        self.assertTrue(EK.check())


class WalshConjugation(unittest.TestCase):
    def test_multidimensional_kernel(self):
        for m in range(2, 6):
            for z in range(1, 2 ** m):
                self.assertTrue(EK.walsh_conjugation_holds(m, z), f"m={m} z={z}")


if __name__ == "__main__":
    unittest.main()
