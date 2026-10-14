import os
import sys
import unittest
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import network_small as NS  # noqa: E402


class NetworkSmall(unittest.TestCase):
    def test_identity_RG_plus_JV(self):
        for h in [5, 6, 7]:
            self.assertTrue(NS.check_identity(h), f"h={h}")

    def test_invocation_adds_x_to_y_and_restores(self):
        for h in [5, 6, 7]:
            self.assertTrue(NS.network_check(h, seed=h), f"h={h}")

    def test_neighbour_count_matches_paper(self):
        self.assertEqual(NS.d_nbr(100), 147731)
        for h in [5, 6, 7, 8]:
            self.assertEqual(NS.d_nbr(h), comb(h - 3, 3) + 3 * (h - 3))

    def test_three_stage_composition_full_grid(self):
        for h in [5, 6]:
            self.assertTrue(NS.three_stage_check(h), f"h={h}")

    def test_wire_count_matches_paper(self):
        h = 100
        v = comb(h, 3)
        N0, I0 = v ** 3, 3 * v * v
        d = NS.d_nbr(h)
        self.assertEqual(2 * N0 + I0 * (v * d + 101), 1873807244643542670000)


class NetworkWiresEnumerated(unittest.TestCase):
    def test_explicit_wire_count_equals_formula(self):
        for h in [5, 6, 7]:
            total, inv, v, dn = NS.enumerate_wires(h)
            W, I0, d = NS.wire_formula(h)
            self.assertEqual(total, W, f"h={h}")
            self.assertEqual(inv, I0, f"h={h}")
            self.assertEqual(dn, d, f"h={h}")


class ResidualSumAndDelta(unittest.TestCase):
    def test_s_and_delta_match_paper(self):
        from math import comb as _comb
        h = 100
        v = _comb(h, 3)
        N0, I0 = v ** 3, 3 * v * v
        W = 1873807244643542670000
        m = 10 ** 6
        s_claim = 1873807244636671267308000000
        delta_claim = 6871402692000000
        self.assertEqual(W * m - 2 * N0 + 2 * I0 * 101 * 100, s_claim)
        self.assertEqual(W * m - s_claim, delta_claim)
        self.assertEqual(2 * N0 - 2 * I0 * 10100, delta_claim)


if __name__ == "__main__":
    unittest.main()
