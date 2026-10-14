import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import conv_dag as CD  # noqa: E402


def depth(dag):
    return max(dag.depths()) if dag.gates else 0


class ConvolutionDAG(unittest.TestCase):
    def test_matches_convolution(self):
        rng = np.random.default_rng(21)
        for m in [1, 2, 3, 5, 8, 13, 31, 64]:
            for k in [1, 2, 3, 7, 16, 33]:
                c = rng.standard_normal(k) + 1j * rng.standard_normal(k)
                x = rng.standard_normal(m) + 1j * rng.standard_normal(m)
                dag = CD.conv_dag(c, m)
                got = np.array(dag.evaluate(list(x)), dtype=np.complex128)
                want = np.convolve(x, c)
                err = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300)
                self.assertLess(err, 1e-10, f"m={m} k={k}")

    def test_gate_count_is_v_log_v(self):
        worst = 0.0
        rng = np.random.default_rng(22)
        for m in [4, 16, 64, 256, 512]:
            for k in [2, 8, 32, 128]:
                c = rng.standard_normal(k) + 0j
                dag = CD.conv_dag(c, m)
                v = m + k - 1
                ratio = len(dag.gates) / (v * math.log2(2 + v))
                worst = max(worst, ratio)
        print(f"\ngates / (v log2(2+v)) worst over sizes: {worst:.3f}")
        self.assertLess(worst, 8.0)

    def test_depth_is_log_v(self):
        worst = 0.0
        rng = np.random.default_rng(23)
        for m in [4, 16, 64, 256]:
            for k in [2, 8, 32]:
                c = rng.standard_normal(k) + 0j
                dag = CD.conv_dag(c, m)
                v = m + k - 1
                worst = max(worst, depth(dag) / math.log2(2 + v))
        print(f"\ndepth / log2(2+v) worst over sizes: {worst:.3f}")
        self.assertLess(worst, 6.0)

    def test_bounded_fan_out(self):
        rng = np.random.default_rng(24)
        worst = 0
        for m in [8, 64, 256]:
            for k in [3, 17, 64]:
                c = rng.standard_normal(k) + 0j
                dag = CD.conv_dag(c, m)
                worst = max(worst, dag.fan_out())
        print(f"\nworst fan-out (including output reads): {worst}")
        self.assertLessEqual(worst, 4)

    def test_predecessors_at_most_two(self):
        c = np.array([1, 2, 3], dtype=complex)
        dag = CD.conv_dag(c, 20)
        for preds in dag.gates:
            self.assertLessEqual(len(preds), 2)


if __name__ == "__main__":
    unittest.main()
