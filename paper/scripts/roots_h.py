"""Roots of h(z) = sum_k h_k z^k for the random Toeplitz inputs of table 3 (article v6, 6.3).

The reciprocal series g = 1/h(z) has radius of convergence rho = min |root of h|, so
|g_k| grows like rho^{-k} (up to polynomial factors). Predicted log10 max_{k<r} |g_k| is about
(r - 1) * log10(1/rho). Roots are computed as the eigenvalues of the companion matrix via
numpy.roots, then the relative residual |h(z)| at the smallest-modulus roots is checked in
mpmath high precision (<= 4e-14 for the widths in the paper).
Usage: python roots_h.py
"""
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import mpmath as mp
import numpy as np
from allength.mp_exact import mp_reciprocal

for r, obs_g, lost in ((352, 28.3, 23.6), (640, 39.4, 35.0), (1024, 88.3, 78.6)):
    rnd = random.Random(r)
    h = []
    for _ in range(r + 1):
        a, b = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
        h.append(complex(a, b))
    h[0] = abs(h[0]) + 1
    # numpy roots: highest degree first; use h_0..h_{r-1} (the width-r matrix) -> degree r-1
    coeffs = np.array(h[:r][::-1], dtype=np.complex128)
    roots = np.roots(coeffs)
    mods = np.sort(np.abs(roots))
    # residual check with mpmath at 60 digits for the smallest roots
    mp.mp.dps = 60
    hp = [mp.mpc(z.real, z.imag) for z in h[:r]]
    def hval(z):
        acc = mp.mpc(0)
        for c in reversed(hp):
            acc = acc * z + c
        return acc
    small = roots[np.argsort(np.abs(roots))[:3]]
    res = [float(abs(hval(mp.mpc(z.real, z.imag))) / sum(abs(c) * abs(z) ** k for k, c in enumerate(hp))) for z in small]
    rho = mods[0]
    pred = (r - 1) * np.log10(1 / rho)
    # actual growth exponent of g fitted from the tail
    mp.mp.dps = 120
    g = mp_reciprocal([mp.mpc(z.real, z.imag) for z in h], r)
    lg = [float(mp.log10(abs(x))) for x in g]
    k0 = r // 2
    slope = (lg[-1] - lg[k0]) / (r - 1 - k0)
    print(f"r={r}: min|root|={rho:.6f} (next {mods[1]:.6f}, {mods[2]:.6f}); roots inside unit disk: "
          f"{int(np.sum(mods < 1))} of {r - 1}; rel. residual at smallest roots <= {max(res):.1e}; "
          f"predicted log10 max|g| ~ {pred:.1f}, observed {max(lg):.1f}; fitted slope {slope:.4f} "
          f"vs log10(1/rho) {np.log10(1 / rho):.4f}; digits lost {lost}", flush=True)
