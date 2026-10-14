"""Scientific regression checks for the phase intervention experiment."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]


class PhasePrecisionImport(unittest.TestCase):
    def test_import_does_not_run_cli(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-c", "from paper.scripts import phase_precision"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "")


class PhasePrecisionArithmetic(unittest.TestCase):
    def test_shear_rounds_multiply_before_add(self):
        from paper.scripts import phase_precision as pp
        with mp.workdps(100):
            coefficient = mp.mpf("1.23456789")
            source = mp.mpf("9.87654321")
            destination = -coefficient * source
            with mp.workdps(10):
                expected = destination + coefficient * source
            result = pp.replay([[(0, 1, coefficient)]], [(2, "chunk")],
                               [source, destination], set(), 10)
            self.assertEqual(result[1], expected)
            self.assertNotEqual(expected, 0)

    def test_T_receives_actual_chain_input(self):
        from paper.scripts import phase_precision as pp
        with mp.workdps(100):
            r = 8
            _, _, stage_input, h, H, x = pp.build_experiment(r, 30, "word")
            u = mp.exp(2j * mp.pi / r)
            transposed = [sum(h[j-i] * H[j] * x[j] for j in range(i, r))
                          for i in range(r)]
            expected = [(-1)**i * u**(i*(i-1)//2) / H[i] * transposed[i]
                        for i in range(r)]
            self.assertLess(max(abs(a-b) for a, b in zip(stage_input, expected)),
                            mp.mpf("1e-90"))


if __name__ == "__main__":
    unittest.main()
