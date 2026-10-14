import os
import random
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from allength import replay as R  # noqa: E402


def state(dag, x, y, z):
    return list(x) + list(y) + list(z)


class ReplayLemma(unittest.TestCase):
    def check_one(self, dag, x, y, z, layered=False):
        e, a = dag.e, dag.a
        M = dag.matrix()
        Mx = [sum(M[i][j] * x[j] for j in range(e)) for i in range(a)]
        v = state(dag, x, y, z)
        if layered:
            v, _ = R.replay_layered(dag, v)
        else:
            v = R.replay_sequential(dag, v)
        xo, yo, zo = v[:e], v[e:e + a], v[e + a:]
        self.assertEqual(xo, list(x), "x must be unchanged")
        self.assertEqual(zo, list(z), "borrowed z must be restored exactly")
        self.assertEqual(yo, [y[i] + Mx[i] for i in range(a)], "y must become y + Mx")

    def test_random_arbitrary_z(self):
        rng = random.Random(11)
        for trial in range(300):
            e, a, g = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 8)
            dag = R.random_dag(rng, e, a, g)
            x = [rng.randint(-9, 9) for _ in range(e)]
            y = [rng.randint(-9, 9) for _ in range(a)]
            z = [rng.randint(-50, 50) for _ in range(g)]   # arbitrary contents
            self.check_one(dag, x, y, z, layered=False)

    def test_correlated_z(self):
        rng = random.Random(12)
        for trial in range(200):
            e, a, g = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 6)
            dag = R.random_dag(rng, e, a, g)
            x = [rng.randint(-9, 9) for _ in range(e)]
            y = [rng.randint(-9, 9) for _ in range(a)]
            # z correlated with x: z_k = (sum of x) * k + 7
            z = [sum(x) * (k + 1) + 7 for k in range(g)]
            self.check_one(dag, x, y, z, layered=False)
            # z equal to the DAG's own gate values (the "natural" initial contents)
            gv = []
            for preds in dag.gates:
                gv.append(sum(c * (x[i] if k == 'x' else gv[i]) for k, i, c in preds))
            self.check_one(dag, x, y, gv, layered=False)

    def test_layered_equals_sequential(self):
        rng = random.Random(13)
        for trial in range(200):
            e, a, g = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 10)
            dag = R.random_dag(rng, e, a, g)
            x = [rng.randint(-9, 9) for _ in range(e)]
            y = [rng.randint(-9, 9) for _ in range(a)]
            z = [rng.randint(-50, 50) for _ in range(g)]
            v1 = R.replay_sequential(dag, state(dag, x, y, z))
            v2, _ = R.replay_layered(dag, state(dag, x, y, z))
            self.assertEqual(v1, v2)
            self.check_one(dag, x, y, z, layered=True)

    def test_shear_and_layer_counts_vs_lemma(self):
        """Lemma claims at most 8g + 2a shears in total and depth O((delta+1)(H+1))."""
        rng = random.Random(14)
        worst_ratio = 0.0
        for trial in range(200):
            e, a, g = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 10)
            dag = R.random_dag(rng, e, a, g)
            fwd, outs = R.build_sweeps(dag, drop_sources=False)
            fwd2, outs2 = R.build_sweeps(dag, drop_sources=True)
            # total shears of the word: 2 sweeps forward + undo, readouts, both rounds
            total = 2 * (len(fwd) + len(fwd2)) + len(outs) + len(outs2)
            bound = 8 * g + 2 * a
            self.assertLessEqual(total, bound, f"e={e} a={a} g={g}")
            worst_ratio = max(worst_ratio, total / bound)
        print(f"\nshear count / (8g+2a) worst over 200 DAGs: {worst_ratio:.3f}")

    def test_layers_are_disjoint_matchings(self):
        rng = random.Random(15)
        for trial in range(100):
            e, a, g = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 10)
            dag = R.random_dag(rng, e, a, g)
            fwd, _ = R.build_sweeps(dag, drop_sources=False)
            depths = dag.depths()
            e_, a_ = dag.e, dag.a
            layers = R.colour_layers(fwd, lambda dst: 0 if dst < e_ + a_ else depths[dst - e_ - a_])
            for L in layers:
                coords = [c for s in L for c in (s[0], s[1])]
                self.assertEqual(len(coords), len(set(coords)), "layer is not a matching")

    def test_depth_scales_with_delta_and_H(self):
        """Layer count grows at most linearly in (delta+1)*(H+1) for these random DAGs."""
        rng = random.Random(16)
        worst = 0.0
        for trial in range(150):
            e, a, g = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 12)
            dag = R.random_dag(rng, e, a, g)
            x = [0] * e
            y = [0] * a
            z = [0] * g
            _, layers = R.replay_layered(dag, state(dag, x, y, z))
            H = max(dag.depths()) if dag.g else 0
            delta = dag.fan_out()
            ratio = layers / ((delta + 1) * (H + 1))
            worst = max(worst, ratio)
        print(f"\nlayers / ((delta+1)(H+1)) worst over 150 DAGs: {worst:.3f}")
        self.assertLess(worst, 10.0)


if __name__ == "__main__":
    unittest.main()
