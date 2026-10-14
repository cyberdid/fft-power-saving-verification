"""Step 24: scalar identities of the directional kernel (network.tex, lemma net:edge).

a = (1+i)/2, b = (1-i)/2, C = a*I + b*R with R = swap (R^2 = I).
Walsh form:  i^{[s]} = a + b(-1)^s  for s in {0,1}  (so the weight-1 phase kernel is C).
For odd weight w: i^{w[s]} = a_w + b_w(-1)^s with (a_w, b_w) = (a,b) if w = 1 mod 4 and the conjugate
pair if w = 3 mod 4. Also C^2 = R and C^{-1} = R C.
"""

import numpy as np

a = (1 + 1j) / 2
b = (1 - 1j) / 2
I2 = np.eye(2, dtype=complex)
R = np.array([[0, 1], [1, 0]], dtype=complex)
C = a * I2 + b * R


def check():
    ok = True
    for s in (0, 1):
        ok &= np.isclose(1j ** s, a + b * (-1) ** s)
    for w in (1, 3, 5, 7, 9, 11):
        aw, bw = (a, b) if w % 4 == 1 else (np.conj(a), np.conj(b))
        for s in (0, 1):
            ok &= np.isclose(1j ** (w * s), aw + bw * (-1) ** s)
    ok &= np.allclose(R @ R, I2)
    ok &= np.allclose(C @ C, R)
    ok &= np.allclose(np.linalg.inv(C), R @ C)
    return bool(ok)


def hadamard(m):
    N = 2 ** m
    H = np.array([[(-1) ** bin(x & y).count('1') for y in range(N)] for x in range(N)],
                 dtype=complex) / np.sqrt(N)
    return H


def walsh_conjugation_holds(m, z):
    """P_z = diag(i^{[z.x]}) equals H (a I + b T_z) H with T_z the translation x -> x xor z."""
    N = 2 ** m
    H = hadamard(m)
    P = np.diag([1j ** (bin(z & x).count('1') % 2) for x in range(N)])
    T = np.zeros((N, N), dtype=complex)
    for x in range(N):
        T[x, x ^ z] = 1          # (T f)(x) = f(x xor z)
    return bool(np.allclose(P, H @ (a * np.eye(N) + b * T) @ H))
