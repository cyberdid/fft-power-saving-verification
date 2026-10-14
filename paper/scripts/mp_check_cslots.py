"""Full DFT through the zero-free C-slot pattern, streamed (article v6, table 1c).

Usage:  python mp_check_cslots.py R DPS [NVEC] [--basis] [--conj-dag]

Every shear y_d += mu y_s of the strict word is replaced, with no test for zero, by the two
shears t1 = kappa, t2 = mu - kappa (kappa = 1 + mu * conj(mu)), and each of them by the slot
sequence D3, C, D2, C, D1, C, D0 on the ordered pair (y_s, y_d) (mp_compiler.mp_shear_factors).
Coordinates outside the pair are untouched (identity singletons in every slot), so the blocks
are applied sparsely instead of being stored; the arithmetic is the same as
mp_compiler.mp_apply_blocks(mp_zero_free_blocks(...)).

--conj-dag : conj(mu) is NOT taken from mu. The whole word is built a second time with every
             root replaced by its inverse (u -> 1/u, power-of-two roots conjugated:
             mp_exact.ROOT_SIGN = -1), as in local.tex; its coefficients give conj(mu).
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import mpmath as mp  # noqa: E402

from allength import conv_dag as CD  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True

from allength import mp_compiler as MC  # noqa: E402
from allength import mp_exact as ME  # noqa: E402
from allength.compiler import transpose_word  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
R, DPS = int(args[0]), int(args[1])
NVEC = int(args[2]) if len(args) > 2 else 3
mp.mp.dps = DPS
CA, CB = MC.C_MP[0][0], MC.C_MP[0][1]


def fourier_parts(u):
    H = [mp.mpc(1)]
    for j in range(1, R + 1):
        H.append(H[-1] * (1 - u ** j))
    cc = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(R)]
    h = [1 / H[j] for j in range(R)] + [mp.mpc(0)]
    return H[:R], [cc[k] / H[k] for k in range(R)], h


def c_pair(a, b):
    return CA * a + CB * b, CB * a + CA * b


def apply_shear_slots(y, s, d, t, stats):
    (p0, q0), (p1, q1), (p2, q2), (p3, q3) = MC.mp_shear_factors(t)
    a, b = y[s], y[d]
    a, b = p3 * a, q3 * b
    a, b = c_pair(a, b)
    a, b = p2 * a, q2 * b
    a, b = c_pair(a, b)
    a, b = p1 * a, q1 * b
    a, b = c_pair(a, b)
    a, b = p0 * a, q0 * b
    y[s], y[d] = a, b
    stats["blocks"] += 1


def apply_word_cslots(y, word, word_bar, stats):
    for L, Lb in zip(word, word_bar):
        for (s, d, c), (sb, db, cb) in zip(L, Lb):
            if s is None:
                y[d] = c * y[d]
                continue
            mu_bar = cb if word_bar is not word else mp.conj(c)
            kappa = 1 + c * mu_bar
            for t in (kappa, c - kappa):
                apply_shear_slots(y, s, d, t, stats)
    return y


t0 = time.time()
u = mp.exp(2j * mp.pi / R)
Dh, Dmid, h = fourier_parts(u)
word = MC.mp_block_word_local(h, R)
tword = transpose_word(word)
conj_info = ""
if "--conj-dag" in flags:
    ME.ROOT_SIGN = -1
    try:
        _, _, hb = fourier_parts(1 / u)
        word_b = MC.mp_block_word_local(hb, R)
    finally:
        ME.ROOT_SIGN = 1
    same = all(len(L) == len(Lb) and all((s, d) == (sb, db) for (s, d, _), (sb, db, _) in zip(L, Lb))
               for L, Lb in zip(word, word_b)) and len(word) == len(word_b)
    dev = max((abs(cb - mp.conj(c)) / max(1, abs(c)) for L, Lb in zip(word, word_b)
               for (_, _, c), (_, _, cb) in zip(L, Lb)), default=mp.mpf(0))
    conj_info = f" conj-DAG: same structure={same}, max rel |mubar_DAG - conj(mu)|={mp.nstr(dev, 3)}"
    tword_b = transpose_word(word_b)
else:
    word_b, tword_b = word, tword
t_build = time.time() - t0

if R <= 32 or "--basis" in flags:
    xs = [[mp.mpc(1) if t == j else mp.mpc(0) for t in range(R)] for j in range(R)]
    kind = "basis"
else:
    rnd = random.Random(R)
    xs = [[mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1)) for _ in range(R)] for _ in range(NVEC)]
    kind = "random"
upow = [u ** k for k in range(R)]
errs = []
stats = {"blocks": 0}
for x in xs:
    stats["blocks"] = 0
    y = [Dh[i] * x[i] for i in range(R)]
    y = apply_word_cslots(y, tword, tword_b, stats)
    y = [Dmid[i] * y[i] for i in range(R)]
    y = apply_word_cslots(y, word, word_b, stats)
    y = [Dh[i] * y[i] for i in range(R)]
    want = [sum(upow[(i * j) % R] * x[j] for j in range(R)) for i in range(R)]
    num = mp.sqrt(sum(abs(y[i] - want[i]) ** 2 for i in range(R)))
    den = mp.sqrt(sum(abs(w) ** 2 for w in want))
    errs.append(num / den)
mx = max(errs)
print(f"cslots r={R} dps={DPS} {kind}({len(xs)}): shear blocks per vector={stats['blocks']} "
      f"(each 4 monomial + 3 C slots) max rel err={mp.nstr(mx, 3)} "
      f"digits lost={float(DPS + mp.log10(mx)):.1f}{conj_info} "
      f"build={t_build:.0f}s total={time.time() - t0:.0f}s", flush=True)
