import os
import random
import sys
import unittest

import mpmath as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import mp_compiler as MC  # noqa: E402


class MpCompiledFourier(unittest.TestCase):
    def check(self, r, seed):
        mp.mp.dps = 60
        rnd = random.Random(seed)
        rc = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        u = mp.exp(2j * mp.pi / r)
        blocks = MC.mp_compile_blocks(u, r)
        x = [rc() for _ in range(r)]
        got = MC.mp_apply_blocks(blocks, x)
        want = [sum(u ** (i * j) * x[j] for j in range(r)) for i in range(r)]
        num = mp.sqrt(sum(abs(got[i] - want[i]) ** 2 for i in range(r)))
        den = mp.sqrt(sum(abs(w) ** 2 for w in want))
        return num / den

    def test_r2_r3_r4(self):
        for r, s in [(2, 1), (3, 2), (4, 3)]:
            self.assertLess(self.check(r, s), mp.mpf(10) ** -50, f"r={r}")

    def test_r8(self):
        self.assertLess(self.check(8, 4), mp.mpf(10) ** -50)


class MpZeroFree(unittest.TestCase):
    def test_zero_free_word_exact_at_60_digits(self):
        mp.mp.dps = 60
        rnd = random.Random(9)
        for r in [2, 3, 4, 8]:
            rc = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
            u = mp.exp(-2j * mp.pi / r)
            blocks = MC.mp_zero_free_blocks(u, r)
            x = [rc() for _ in range(r)]
            got = MC.mp_apply_blocks(blocks, x)
            want = [sum(u ** (i * j) * x[j] for j in range(r)) for i in range(r)]
            num = mp.sqrt(sum(abs(got[i] - want[i]) ** 2 for i in range(r)))
            den = mp.sqrt(sum(abs(w) ** 2 for w in want))
            self.assertLess(num / den, mp.mpf(10) ** -50, f"r={r}")


class ExactRootsAndStrictTopology(unittest.TestCase):
    def test_both_mp_compilers_accept_exact_roots(self):
        with mp.workdps(80):
            for r,u in ((2,mp.mpc(-1)),(4,mp.j)):
                for compile_word in (MC.mp_compile_blocks,MC.mp_zero_free_blocks):
                    compiled=compile_word(u,r)
                    for j in range(r):
                        x=[mp.mpc(int(k==j)) for k in range(r)]
                        got=MC.mp_apply_blocks(compiled,x)
                        self.assertLess(max(abs(got[k]-u**(k*j)) for k in range(r)),mp.mpf('1e-70'))

    def test_mp_zero_free_topology_restores_global_mode(self):
        from allength import conv_dag as cd
        with mp.workdps(80):
            previous=cd.KEEP_ZERO_MULTIPLIERS
            try:
                signatures=[]
                for flag in (False,True):
                    cd.KEEP_ZERO_MULTIPLIERS=flag
                    compiled=MC.mp_zero_free_blocks(mp.exp(2j*mp.pi/16),16)
                    signatures.append(tuple(tuple(tuple(pairs) for pairs in block['pairs']) for block in compiled))
                    self.assertEqual(cd.KEEP_ZERO_MULTIPLIERS,flag)
                self.assertEqual(signatures[0],signatures[1])
            finally:cd.KEEP_ZERO_MULTIPLIERS=previous


if __name__ == "__main__":
    unittest.main()
