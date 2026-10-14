"""Full DFT check of the compiled word, strict mode (article v5, table 1b).

Usage:  python mp_check_dft.py R DPS [NVEC] [--float] [--cslots]

Checks the whole chain  x -> Dh T Dmid T^T Dh x = F_r(u) x  with u = exp(2 pi i / R), where
T is applied by the strict block word (no gate dropped for a zero coefficient, no tolerance).
  * R <= 32: every basis vector (the whole matrix of the implemented operator) is compared.
  * otherwise NVEC random vectors (real and imaginary parts uniform on [-1, 1], seed R).
--float   : same chain in float64 (numpy word), for comparison.
--cslots  : also run the zero-free C-slot blocks (mp_compiler.mp_zero_free_blocks), R <= 64.
Printed: relative error per vector (max and median), log10 of max |H_k| and max |1/H_k|,
and the number of decimal digits lost (DPS + log10(max error)).
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from allength import conv_dag as CD  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True

from allength import mp_compiler as MC  # noqa: E402
from allength import rounds as RD  # noqa: E402
from allength.compiler import transpose_word  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
R = int(args[0])
DPS = int(args[1])
NVEC = int(args[2]) if len(args) > 2 else 4
mp.mp.dps = DPS


def apply_word(y, word):
    for L in word:
        for s, d, c in L:
            if s is None:
                y[d] = c * y[d]
            else:
                y[d] = y[d] + c * y[s]
    return y


def vectors(r, real):
    if r <= 32 or "--basis" in flags:
        out = []
        for j in range(r):
            e = [0] * r
            e[j] = 1
            out.append(e)
        return out, "basis"
    rnd = random.Random(r)
    return [[complex(rnd.uniform(-1, 1), rnd.uniform(-1, 1)) for _ in range(r)]
            for _ in range(NVEC)], "random"


def run_mp():
    t = time.time()
    u = mp.exp(2j * mp.pi / R)
    H = [mp.mpc(1)]
    for j in range(1, R + 1):
        H.append(H[-1] * (1 - u ** j))
    cc = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(R)]
    h = [1 / H[j] for j in range(R)] + [mp.mpc(0)]
    Dh = [H[i] for i in range(R)]
    Dmid = [cc[k] / H[k] for k in range(R)]
    word = MC.mp_block_word_local(h, R)
    tword = transpose_word(word)
    upow = [u ** k for k in range(R)]
    t_build = time.time() - t
    errs = []
    xs, kind = vectors(R, False)
    for x in xs:
        x = [mp.mpc(v) for v in x]
        y = [Dh[i] * x[i] for i in range(R)]
        y = apply_word(y, tword)
        y = [Dmid[i] * y[i] for i in range(R)]
        y = apply_word(y, word)
        y = [Dh[i] * y[i] for i in range(R)]
        want = [sum(upow[(i * j) % R] * x[j] for j in range(R)) for i in range(R)]
        num = mp.sqrt(sum(abs(y[i] - want[i]) ** 2 for i in range(R)))
        den = mp.sqrt(sum(abs(w) ** 2 for w in want))
        errs.append(num / den)
    errs.sort()
    mx = errs[-1]
    lgH = max(mp.log10(abs(v)) for v in H[:R])
    lgh = max(mp.log10(abs(v)) for v in h[:R])
    print(f"mp  r={R} dps={DPS} {kind}({len(xs)}): max rel err={mp.nstr(mx, 3)} "
          f"median={mp.nstr(errs[len(errs) // 2], 3)} digits lost={float(DPS + mp.log10(mx)):.1f} "
          f"log10 max|H|={float(lgH):.1f} log10 max|1/H|={float(lgh):.1f} "
          f"layers={len(word)} build={t_build:.0f}s total={time.time() - t:.0f}s", flush=True)
    if "--cslots" in flags and R <= 64:
        t = time.time()
        blocks = MC.mp_zero_free_blocks(u, R)
        e2 = []
        for x in xs:
            x = [mp.mpc(v) for v in x]
            y = MC.mp_apply_blocks(blocks, x)
            want = [sum(upow[(i * j) % R] * x[j] for j in range(R)) for i in range(R)]
            num = mp.sqrt(sum(abs(y[i] - want[i]) ** 2 for i in range(R)))
            den = mp.sqrt(sum(abs(w) ** 2 for w in want))
            e2.append(num / den)
        print(f"mp  r={R} dps={DPS} C-slot zero-free blocks={len(blocks)}: "
              f"max rel err={mp.nstr(max(e2), 3)} ({time.time() - t:.0f}s)", flush=True)


def run_float():
    u = np.exp(2j * np.pi / R)
    H = [1 + 0j]
    for j in range(1, R + 1):
        H.append(H[-1] * (1 - u ** j))
    cc = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(R)]
    h = np.array([1 / H[j] for j in range(R)] + [0j], dtype=np.complex128)
    Dh = np.array(H[:R])
    Dmid = np.array([cc[k] / H[k] for k in range(R)])
    word = RD.block_word_local(h, R)
    tword = transpose_word(word)
    F = np.array([[u ** ((i * j) % R) for j in range(R)] for i in range(R)])
    xs, kind = vectors(R, True)
    errs = []
    for x in xs:
        x = np.array(x, dtype=np.complex128)
        y = Dh * x
        y = apply_word(list(y), tword)
        y = Dmid * np.array(y)
        y = apply_word(list(y), word)
        y = Dh * np.array(y)
        want = F @ x
        errs.append(np.linalg.norm(y - want) / np.linalg.norm(want))
    print(f"f64 r={R} {kind}({len(xs)}): max rel err={max(errs):.3g} median={np.median(errs):.3g}",
          flush=True)


if "--float" in flags:
    run_float()
else:
    run_mp()
