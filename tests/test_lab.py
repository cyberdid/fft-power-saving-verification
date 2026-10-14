import unittest
from fractions import Fraction

import numpy as np

import exact
import numerical
import signals


class ExactChecks(unittest.TestCase):
    def test_c_squared_is_exchange_not_identity(self):
        # C^2 = [[0,1],[1,0]]; catches phase/sign errors in the paper kernel.
        self.assertEqual(exact.c_pair(*exact.c_pair(exact.Q(2, 3), exact.Q(-4, 1))),
                         (exact.Q(-4, 1), exact.Q(2, 3)))

    def test_directional_inverse_restores_input(self):
        x = [exact.Q(i, 2-i) for i in range(8)]
        y = exact.directional(x, 3)
        self.assertEqual(exact.directional(y, 3, inverse=True), x)
        with self.assertRaises(ValueError):
            exact.directional(x, 0)

    def test_tensor_two_bits_has_hand_derived_impulse(self):
        self.assertEqual(exact.tensor_c([exact.Q(1), exact.Q(), exact.Q(), exact.Q()]),
                         [exact.Q(0, Fraction(1, 2)), exact.Q(Fraction(1, 2)),
                          exact.Q(Fraction(1, 2)), exact.Q(0, Fraction(-1, 2))])

    def test_scalar_network_restores_dirty_auxiliaries(self):
        net = exact.ScalarNetwork(5)
        x = [exact.Q(i-4, i) for i in range(net.v)]
        y = [exact.Q(3-i, -i) for i in range(net.v)]
        side = [exact.Q(i % 7, -i % 3) for i in range(len(net.edges))]
        center = [exact.Q(i+1, -i-2) for i in range(net.h+1)]
        observed = net.invoke(x, y, side, center)
        self.assertEqual(observed, (x, [a+b for a, b in zip(y, x)], side, center))

    def test_exact_newton_dft_four_sign_and_transpose(self):
        x = [exact.Q(1), exact.Q(2), exact.Q(3), exact.Q(4)]
        self.assertEqual(exact.newton_dft(x),
                         [exact.Q(10), exact.Q(-2, 2), exact.Q(-2), exact.Q(-2, -2)])

    def test_paper_integer_budget_matches_published_integers(self):
        b = exact.network_budget(100)
        self.assertEqual(b['W'], 1873807244643542670000)
        self.assertEqual(b['s'], 1873807244636671267308000000)
        self.assertEqual(b['delta'], 6871402692000000)
        self.assertEqual(b['W_star'], 2**71)
        self.assertEqual(b['K'], 72000000)


class NumericChecks(unittest.TestCase):
    def test_radix_fft_literal_four(self):
        y = numerical.radix2_fft(np.array([1, 2, 3, 4], dtype=complex))
        np.testing.assert_allclose(y, [10, -2+2j, -2, -2-2j], atol=1e-12)
        with self.assertRaises(ValueError):
            numerical.radix2_fft(np.ones(3))

    def test_newton_matches_direct_at_composite_non_power_two_length(self):
        x = np.array([1+2j, -3, 4j, 2, 5-1j, -2j])
        p = numerical.NewtonPlan(len(x))
        np.testing.assert_allclose(p.apply(x), numerical.DirectPlan(len(x)).apply(x),
                                   rtol=1e-10, atol=1e-10)

    def test_signal_generation_is_reproducible_and_has_known_tone_bin(self):
        a = signals.make_cases(64, 7)
        b = signals.make_cases(64, 7)
        np.testing.assert_array_equal(a['white_noise'], b['white_noise'])
        self.assertEqual(np.argmax(np.abs(np.fft.fft(a['bin_tone']))), 7)

    def test_hann_does_not_erase_the_suite_impulse(self):
        x = signals.make_cases(64)['impulse']*signals.hann(64)
        self.assertEqual(float(np.sum(np.abs(x)**2)), 1.0)

    def test_walsh_frame_has_correct_directional_phase(self):
        values = [exact.Q(i-3, 2*i) for i in range(16)]
        self.assertEqual(exact.frame_transform(values, [7]),
                         exact.directional(values, 7, inverse=True))
        self.assertEqual(exact.frame_transform(values, [1, 2, 4, 8]),
                         exact.tensor_c(values))

    def test_periodogram_integrates_to_window_corrected_power(self):
        x = np.ones(16, dtype=complex)
        w = signals.hann(16)
        p = signals.periodogram(np.fft.fft(x*w), 20e6, w)
        self.assertAlmostEqual(float(p.sum()*20e6/16), 1.0, places=12)

    def test_equal_outputs_have_zero_error_and_invalid_input_is_rejected(self):
        y = np.array([1j, 2, -3j, 0])
        m = signals.compare(y, y, 20e6, np.ones(4))
        self.assertEqual(m['relative_l2_error'], 0)
        self.assertEqual(m['psd_max_delta_db'], 0)
        with self.assertRaises(ValueError):
            numerical.DirectPlan(0)


if __name__ == '__main__':
    unittest.main()
