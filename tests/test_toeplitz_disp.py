import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import toeplitz_disp as TD  # noqa: E402


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


class ToeplitzDisplacement(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(31)

    def rand_h(self, n):
        h = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        h[0] = abs(h[0]) + 1.0  # nonzero diagonal
        return h

    def test_reciprocal_inverse(self):
        for v in [1, 2, 5, 16, 33]:
            h = self.rand_h(v)
            g = TD.reciprocal_series(h, v)
            T = TD.toeplitz_matrix(h, v)
            Ti = TD.toeplitz_matrix(g, v)
            self.assertLess(rel(T @ Ti, np.eye(v)), 1e-9, f"v={v}")

    def test_cross_block_factorisation(self):
        # T_v = [[I,0],[E,I]] diag(T_p, T_{v-p}) with E = B T_p^{-1}
        for v in [2, 3, 6, 9, 16]:
            h = self.rand_h(v)
            p = v // 2
            g = TD.reciprocal_series(h, v)
            T = TD.toeplitz_matrix(h, v)
            Tp = TD.toeplitz_matrix(h, p)
            Tq = TD.toeplitz_matrix(h, v - p)
            E = np.array([[TD.cross_entry(h, g, p, i, j) for j in range(p)] for i in range(p, v)])
            B = T[p:, :p]
            self.assertLess(rel(E, B @ np.linalg.inv(Tp)), 1e-9, f"v={v} E formula")
            F = np.block([[np.eye(p), np.zeros((p, v - p))],
                          [E, np.eye(v - p)]]) @ np.block([[Tp, np.zeros((p, v - p))],
                                                           [np.zeros((v - p, p)), Tq]])
            self.assertLess(rel(F, T), 1e-9, f"v={v} factorisation")

    def test_displacement_entries(self):
        for v in [8, 13, 20]:
            h = self.rand_h(v)
            p = v // 2
            g = TD.reciprocal_series(h, v)
            for i in range(p + 1, v):
                for j in range(1, p):
                    lhs = TD.cross_entry(h, g, p, i, j) - TD.cross_entry(h, g, p, i - 1, j - 1)
                    rhs = -h[i - p] * g[p - j]
                    self.assertLess(abs(lhs - rhs), 1e-9 * max(1, abs(rhs)), f"v={v} i={i} j={j}")

    def test_chunk_reconstruction_and_outer_apply(self):
        # M_chunk = sum_l J^l D (J^T)^l, D = v1 w1^T + e0 R0^T + u e0^T  (checked densely)
        for v in [9, 16, 21]:
            h = self.rand_h(v)
            p = v // 2
            g = TD.reciprocal_series(h, v)
            for chunk in [2, 3, 5]:
                for i0 in range(p + 1, v, chunk):
                    a = min(chunk, v - i0)
                    for j0 in range(1, p, chunk):
                        e = min(chunk, p - j0)
                        M = np.array([[TD.cross_entry(h, g, p, i0 + r, j0 + c) for c in range(e)]
                                      for r in range(a)])
                        parts = TD.chunk_displacement(h, g, p, i0, a, j0, e)
                        D = sum(np.outer(vv, ww) for vv, ww in parts)
                        D_direct = M.copy()
                        D_direct[1:, 1:] = M[1:, 1:] - M[:-1, :-1]
                        self.assertLess(rel(D, D_direct), 1e-9, f"D v={v} chunk={chunk}")
                        # dense reconstruction sum_l J^l D (J^T)^l
                        Rec = np.zeros((a, e), dtype=np.complex128)
                        Jl = np.eye(a, dtype=np.complex128)
                        Jr = np.eye(e, dtype=np.complex128)
                        for l in range(max(a, e)):
                            Rec += Jl @ D @ Jr.T
                            Jl = np.vstack([np.zeros((1, a)), Jl[:-1]]) if l >= 0 else Jl
                            Jr = np.vstack([np.zeros((1, e)), Jr[:-1]])
                        # shift matrices: J^l applies to rows, J_e^T^l to columns (lower shifts)
                        self.assertLess(rel(Rec, M), 1e-8, f"rec v={v} chunk={chunk}")
                        # application through the two-convolution path
                        x = self.rng.standard_normal(e) + 1j * self.rng.standard_normal(e)
                        got = sum(TD.outer_apply(vv, ww, x, a) for vv, ww in parts)
                        self.assertLess(rel(got, M @ x), 1e-8, f"apply v={v} chunk={chunk}")

    def test_recursive_toeplitz_apply(self):
        for n in [1, 2, 3, 4, 7, 8, 16, 25, 32, 50, 64]:
            h = self.rand_h(n)
            x = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
            want = TD.toeplitz_matrix(h, n) @ x
            for chunk in [1, 2, 4, 8]:
                got = TD.toeplitz_apply(h, x, chunk=chunk)
                self.assertLess(rel(got, want), 1e-8, f"n={n} chunk={chunk}")


if __name__ == "__main__":
    unittest.main()
