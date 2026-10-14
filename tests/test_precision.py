import unittest

import mpmath as mp

from precision_control import direct, prepare, relative_error, transform


class PrecisionChecks(unittest.TestCase):
    def test_literal_four_and_larger_identity(self):
        with mp.workdps(50):
            roots, N = prepare(4)
            actual = transform([mp.mpc(i) for i in (1, 2, 3, 4)], N)
            expected = [mp.mpc(10), mp.mpc(-2, 2), mp.mpc(-2), mp.mpc(-2, -2)]
            self.assertLess(relative_error(actual, expected), mp.mpf('1e-45'))
            roots, N = prepare(8)
            values = [mp.mpc(i-3, 2*i-1) for i in range(8)]
            self.assertLess(relative_error(transform(values, N), direct(values, roots)), mp.mpf('1e-45'))


if __name__ == '__main__':
    unittest.main()
