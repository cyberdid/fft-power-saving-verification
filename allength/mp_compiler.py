"""Step 12: the uniform local compiler for F_r(u) in arbitrary precision (mpmath).

Same construction as allength/compiler.py (Newton reduction F = Dh T diag(c_k/H_k) T^T Dh,
block-local Toeplitz word, shears replaced by D0 C D1 C D2 C D3 with closed-form D's, fixed
slot pattern), but every scalar is mp. Purpose: separate the algebra from float64 rounding.
With dps=60 the float noise (~1e-17) becomes ~1e-60, so the zero threshold (1e-30) is safe and
the compiled word should equal F_r(u) to roughly the working precision.
"""

import mpmath as mp

from allength.chunk_dag import compose  # noqa: F401  (generic; kept for readers)
from allength.mp_exact import mp_chunk_parts, mp_chunk_dag, mp_reciprocal
from allength.rounds import admissible_chunk, replay_layers  # noqa: F401
from allength.compiler import transpose_word  # generic (reverses order, swaps shear ends)
from allength.conv_dag import zero_multiplier_mode

ZERO_TOL_MP = mp.mpf(10) ** -30


def mp_C():
    a = (1 + 1j) / 2
    b = (1 - 1j) / 2
    return [[mp.mpc(a), mp.mpc(b)], [mp.mpc(b), mp.mpc(a)]]


C_MP = mp_C()


def mp_shear_factors(c):
    c = mp.mpc(c)
    return [(-5j / (3 * c), mp.mpf(1)), (mp.mpf(2), mp.mpf(1)), (mp.mpf(3), mp.mpf(-1)),
            (2 * c / (5 * (1 + 1j)), 1 / (2 * (1 - 1j)))]


def mp_block_word_local(h, r):
    """Block-local exact-width word, mp arithmetic (mirrors rounds.block_word_local)."""
    g = mp_reciprocal(h, r + 1)
    word = []

    def rec(lo, hi):
        v = hi - lo
        if v == 1:
            word.append([(None, lo, h[0])])
            return
        b = admissible_chunk(v)
        if b == 0:
            for i in range(v - 1, -1, -1):
                word.append([(None, lo + i, h[0])])
                for j in range(i):
                    word.append([(lo + j, lo + i, h[i - j])])
            return
        p = v // 2
        rec(lo, lo + p)
        rec(lo + p, hi)
        for i0 in range(lo + p, hi, b):
            I = list(range(i0, min(i0 + b, hi)))
            for j0 in range(lo, lo + p, b):
                J = list(range(j0, min(j0 + b, lo + p)))
                parts = mp_chunk_parts(h, g, p, i0 - lo, len(I), j0 - lo, len(J))
                dag = mp_chunk_dag(parts, len(J))
                used = set(I) | set(J)
                borrowed = [q for q in range(lo, hi) if q not in used]
                Z = borrowed[:dag.g]
                to_global = list(J) + list(I) + list(Z)
                for L in replay_layers(dag):
                    word.append([(to_global[s] if s is not None else None, to_global[d], c)
                                 for s, d, c in L])
    rec(0, r)
    return word


def mp_compile_blocks(u, r):
    """Fixed-pattern blocks for F_r(u) in mp. Each block: 4 diagonals (lists of length r),
    3 pair-lists for the C slots."""
    u = mp.mpc(u)
    H = [mp.mpf(1) + 0j]
    for j in range(1, r + 1):
        H.append(H[-1] * (1 - u ** j))
    cc_ = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(r)]
    # H_r = 0 for a primitive r-th root; h_r is padding (unused by the width-r word), set to 0.
    h = [1 / H[j] for j in range(r)] + [mp.mpc(0)]
    Dh = [H[i] for i in range(r)]
    Dmid = [cc_[k] / H[k] for k in range(r)]
    word = mp_block_word_local(h, r)
    tword = transpose_word(word)

    def ident():
        return {'diag': [[mp.mpc(1)] * r for _ in range(4)], 'pairs': [[], [], []]}

    def diag_block(vec):
        blk = ident()
        blk['diag'][0] = list(vec)
        return blk

    def layer_block(layer):
        blk = ident()
        shears = [(s, d, c) for s, d, c in layer if s is not None and abs(c) >= ZERO_TOL_MP]
        scales = [(d, c) for s, d, c in layer if s is None]
        for k in range(4):
            vec = blk['diag'][k]
            for s, d, c in shears:
                p = mp_shear_factors(c)[3 - k]
                vec[s] = p[0]
                vec[d] = p[1]
            if k == 0:
                for d, c in scales:
                    vec[d] = vec[d] * c
        if shears:
            blk['pairs'] = [[(s, d) for s, d, _ in shears]] * 3
        return blk

    blocks = [diag_block(Dh)]
    blocks += [layer_block(L) for L in tword]
    blocks.append(diag_block(Dmid))
    blocks += [layer_block(L) for L in word]
    blocks.append(diag_block(Dh))
    return blocks


def mp_apply_blocks(blocks, x):
    y = [mp.mpc(v) for v in x]
    for blk in blocks:
        y = [blk['diag'][0][i] * y[i] for i in range(len(y))]
        for k in range(3):
            for s, d in blk['pairs'][k]:
                a0, a1 = y[s], y[d]
                y[s] = C_MP[0][0] * a0 + C_MP[0][1] * a1
                y[d] = C_MP[1][0] * a0 + C_MP[1][1] * a1
            y = [blk['diag'][k + 1][i] * y[i] for i in range(len(y))]
    return y


# ---- zero-free compilation in mp (step 17b) -----------------------------------------------
def mp_split_shear(mu):
    mu = mp.mpc(mu)
    kappa = 1 + abs(mu) ** 2
    return kappa, mu - kappa


def mp_zero_free_blocks(u, r, *, strict=True):
    """Zero-free blocks; strict topology by default, legacy pruning only with strict=False."""
    u = mp.mpc(u)
    H = [mp.mpc(1)]
    for j in range(1, r + 1):
        H.append(H[-1] * (1 - u ** j))
    cc_ = [(-1) ** k * u ** (k * (k - 1) // 2) for k in range(r)]
    # H_r = 0 for a primitive r-th root; h_r is padding (unused by the width-r word), set to 0.
    h = [1 / H[j] for j in range(r)] + [mp.mpc(0)]
    Dh = [H[i] for i in range(r)]
    Dmid = [cc_[k] / H[k] for k in range(r)]
    with zero_multiplier_mode(strict):
        word = mp_block_word_local(h, r)
    tword = transpose_word(word)

    def ident():
        return {'diag': [[mp.mpc(1)] * r for _ in range(4)], 'pairs': [[], [], []]}

    def diag_block(vec):
        blk = ident()
        blk['diag'][0] = list(vec)
        return blk

    def layer_blocks(layer):
        shears = [(s, d, c) for s, d, c in layer if s is not None]
        scales = [(d, c) for s, d, c in layer if s is None]
        out = []
        for part in (0, 1):
            blk = ident()
            for k in range(4):
                vec = blk['diag'][k]
                for s, d, c in shears:
                    t = mp_split_shear(c)[part]
                    p = mp_shear_factors(t)[3 - k]
                    vec[s] = p[0]
                    vec[d] = p[1]
                if k == 0 and part == 0:
                    for d, c2 in scales:
                        vec[d] = vec[d] * c2
            if shears:
                blk['pairs'] = [[(s, d) for s, d, _ in shears]] * 3
            out.append(blk)
        return out

    blocks = [diag_block(Dh)]
    for L in tword:
        blocks += layer_blocks(L)
    blocks.append(diag_block(Dmid))
    for L in word:
        blocks += layer_blocks(L)
    blocks.append(diag_block(Dh))
    return blocks
