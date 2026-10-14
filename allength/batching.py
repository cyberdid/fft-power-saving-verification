"""Step 22: exact-batching recurrence and the critical exponent (network.tex, theorem net:tensor-bound).

t(k) <= A + lam * t(floor(k/m)) for k >= K, with lam in (1, m); the claim is t(k) = O(k^theta),
theta = log_m(lam). Here the recurrence is evaluated with equality (worst case) for small m and
any lam, and the ratio t(k)/k^theta is reported.
"""

import math


def t_worst(k, m, lam, A=1.0, K=None, B=1.0):
    """Equality case of the recurrence; base case t(j) = B for j < K."""
    if K is None:
        K = m
    if k < K:
        return B
    return A + lam * t_worst(k // m, m, lam, A, K, B)


def growth_ratio(m, lam, kmax=10 ** 6):
    theta = math.log(lam) / math.log(m)
    worst_lo, worst_hi = float("inf"), 0.0
    k = 2 * m
    while k <= kmax:
        r = t_worst(k, m, lam) / (k ** theta)
        worst_lo = min(worst_lo, r)
        worst_hi = max(worst_hi, r)
        k = int(k * 1.7) + 1
    return theta, worst_lo, worst_hi
