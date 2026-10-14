"""Stage-wise error analysis of the DFT chain x -> Dh T Dmid T^T Dh x (article v6, table 2b).

Usage:  python stagewise.py R DPS [--cond]

Stages: 1 = Dh, 2 = T^T (word), 3 = Dmid, 4 = T (word), 5 = Dh.
A reference run at DPS + 150 digits gives the exact stage outputs s_k (to far below DPS).
Reported per stage, for one random vector (seed R):
  local   : digits lost by this stage alone at DPS, started from the reference input s_{k-1}
            (log10 of ||stage_DPS(s_{k-1}) - s_k|| / ||s_k|| plus DPS);
  cumul   : digits lost by the whole chain up to this stage at DPS;
  growth  : log10 of max |intermediate coordinate| during the stage / ||s_{k-1}||_inf;
  norm    : log10 ||s_k||_2 / ||x||_2.
--cond : also log10 cond_2(T) and cond_2(Dh) (mpmath SVD; R <= 64).
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import mpmath as mp  # noqa: E402

from allength import conv_dag as CD  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True

from allength import mp_compiler as MC  # noqa: E402
from allength.compiler import transpose_word  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
R, DPS = int(args[0]), int(args[1])
REF = DPS + 150


def build(dps):
    mp.mp.dps = dps
    u = mp.exp(2j * mp.pi / R)
    H = [mp.mpc(1)]
    for j in range(1, R + 1):
        H.append(H[-1] * (1 - u ** j))
    cc = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(R)]
    h = [1 / H[j] for j in range(R)] + [mp.mpc(0)]
    word = MC.mp_block_word_local(h, R)
    return {"Dh": H[:R], "Dmid": [cc[k] / H[k] for k in range(R)], "word": word,
            "tword": transpose_word(word), "H": H}


def run_stage(P, k, y):
    """Apply stage k to y (list of mpc); returns (output, max |intermediate|)."""
    y = list(y)
    if k in (1, 5):
        out = [P["Dh"][i] * y[i] for i in range(R)]
        return out, max(abs(v) for v in out)
    if k == 3:
        out = [P["Dmid"][i] * y[i] for i in range(R)]
        return out, max(abs(v) for v in out)
    word = P["tword"] if k == 2 else P["word"]
    peak = max(abs(v) for v in y)
    for L in word:
        for s, d, c in L:
            if s is None:
                y[d] = c * y[d]
            else:
                y[d] = y[d] + c * y[s]
            a = abs(y[d])
            if a > peak:
                peak = a
    return y, peak


def norm(v):
    return mp.sqrt(sum(abs(t) ** 2 for t in v))


rnd = random.Random(R)
x_raw = [(rnd.uniform(-1, 1), rnd.uniform(-1, 1)) for _ in range(R)]

ref = build(REF)
mp.mp.dps = REF
xs = [mp.mpc(a, b) for a, b in x_raw]
S = [xs]
peaks = [None]
for k in range(1, 6):
    out, pk = run_stage(ref, k, S[-1])
    S.append(out)
    peaks.append(pk)

low = build(DPS)
names = {1: "Dh", 2: "T^T", 3: "Dmid", 4: "T", 5: "Dh"}
print(f"r={R} dps={DPS} (reference {REF})")
print("stage  local_lost  cumul_lost  log10_growth  log10_norm")
mp.mp.dps = DPS
y = [mp.mpc(a, b) for a, b in x_raw]
for k in range(1, 6):
    mp.mp.dps = DPS
    y, _ = run_stage(low, k, y)
    loc_in = [mp.mpc(v) for v in S[k - 1]]          # reference input rounded to DPS
    loc, _ = run_stage(low, k, loc_in)
    mp.mp.dps = REF
    ek = norm([y[i] - S[k][i] for i in range(R)]) / norm(S[k])
    el = norm([loc[i] - S[k][i] for i in range(R)]) / norm(S[k])
    inf_in = max(abs(v) for v in S[k - 1])
    lg = mp.log10(peaks[k] / inf_in)
    print(f"{k} {names[k]:<5} {float(DPS + mp.log10(el)):10.1f} {float(DPS + mp.log10(ek)):10.1f} "
          f"{float(lg):12.1f} {float(mp.log10(norm(S[k]) / norm(S[0]))):11.1f}", flush=True)

if "--cond" in sys.argv and R <= 64:
    mp.mp.dps = 60 + R
    P = build(60 + R)
    H = P["H"]
    T = mp.matrix(R, R)
    for i in range(R):
        for j in range(i + 1):
            T[i, j] = 1 / H[i - j]
    sv = mp.svd_c(T, compute_uv=False)
    svs = [abs(sv[i]) for i in range(R)]
    print(f"log10 cond2(T) = {float(mp.log10(max(svs) / min(svs))):.1f}; "
          f"log10 cond2(Dh) = {float(mp.log10(max(abs(v) for v in H[:R]) / min(abs(v) for v in H[:R]))):.1f}",
          flush=True)
