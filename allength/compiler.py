"""Step 11: uniform local compiler for F_r(u) (local.tex, Proposition loc:compiler).

Target: F_r(u) = Dh * T * diag(c_k / H_k) * T^T * Dh, where
  H_j = prod_{s=1}^{j} (1 - u^s),  c_k = (-1)^k u^{k(k-1)/2},  Dh = diag(H_i),
  T = lower Toeplitz with entries h_{i-j} = 1/H_{i-j}  (block-local exact-width word, step 10).
(This is the Newton reduction N D_N^{-1} N^T with N = Dh T diag(c_k).)

Slot types of the compiled word (as in the proposition):
  * monomial slot: a diagonal matrix on all r coordinates;
  * C slot: copies of C = [[a, b],[b, a]] (a=(1+i)/2, b=(1-i)/2) on disjoint ordered pairs,
    identity elsewhere.
Every shear  x_dst += c x_src  is replaced by a product  D0 C D1 C D2 C D3  (4 monomial slots,
3 C slots) acting on the pair (src, dst), which is verified numerically for each c.
A scaling by h0 is a monomial slot.

Coefficients for the shear factorisation are a CLOSED FORM rational in c and i (step 11b);
the only division is by c, which is nonzero for every non-identity shear.
"""

import cmath
import math

import numpy as np
from scipy.optimize import least_squares

from allength.rounds import block_word_local
from allength.conv_dag import zero_multiplier_mode

A = (1 + 1j) / 2
B = (1 - 1j) / 2
C = np.array([[A, B], [B, A]], dtype=np.complex128)


def shear_factors(c):
    """Closed form: [[1,0],[c,1]] = D0 C D1 C D2 C D3 with
        D0 = diag(-5i/(3c), 1),  D1 = diag(2, 1),  D2 = diag(3, -1),
        D3 = diag(2c/(5(1+i)), 1/(2(1-i))).
    Derived from H D H = alpha X + beta Z (H = C, X = swap, Z = diag(1,-1)) and the entry
    equations of the product, with the free parameter t = 2 fixed. Valid for c != 0; the only
    division is by c (a nonzero coefficient). Returns [(p0,q0),(p1,q1),(p2,q2),(p3,q3)]."""
    if c == 0:
        raise ValueError("shear with c = 0 is the identity; it needs no factorisation")
    c = complex(c)
    return [(-5j / (3 * c), 1.0), (2.0, 1.0), (3.0, -1.0),
            (2 * c / (5 * (1 + 1j)), 1 / (2 * (1 - 1j)))]


def shear_factor_residual(c):
    """max |D0 C D1 C D2 C D3 - [[1,0],[c,1]]| (used by the tests)."""
    M = np.eye(2, dtype=np.complex128)
    for k, (p, q) in enumerate(shear_factors(c)):
        M = M @ np.diag([p, q])
        if k < 3:
            M = M @ C
    return np.max(np.abs(M - np.array([[1, 0], [c, 1]], dtype=np.complex128)))


def transpose_word(word):
    """Transpose of a word: reverse the layer order; a shear (src,dst,c) becomes (dst,src,c)."""
    out = []
    for L in reversed(word):
        out.append([(dst if src is not None else None, src if src is not None else dst, c)
                    for src, dst, c in L])
    return out


def newton_coefficients(u, r):
    H = [1 + 0j]
    for j in range(1, r + 1):
        H.append(H[-1] * (1 - u ** j))
    c = [(-1) ** k * cmath.exp(2j * math.pi * 0) * u ** (k * (k - 1) // 2) for k in range(r)]
    return H, c


def compile_fourier(u, r):
    """Slot sequence of F_r(u). Returns list of slots: ('diag', vec) or ('C', pairs_list)."""
    H, c = newton_coefficients(u, r)
    # H_r = 0 for a primitive r-th root (1 - u^r = 0), so h_r = 1/H_r is undefined.
    # The width-r word uses only h_0..h_{r-1}; h_r is padding, set to 0 (see exact_check_dft).
    h = [1 / H[j] for j in range(r)] + [0]
    Dh = np.array([H[i] for i in range(r)], dtype=np.complex128)
    Dmid = np.array([c[k] / H[k] for k in range(r)], dtype=np.complex128)
    word = block_word_local(np.array(h, dtype=np.complex128), r)
    tword = transpose_word(word)

    slots = [('diag', Dh)]          # apply order: Dh, T^T, Dmid, T, Dh  (matrix Dh T Dmid T^T Dh)

    def emit_word(w):
        for L in w:
            # DEVIATION (float64 only): shears with |c| < 1e-12 are identities. The paper keeps such
            # gates structurally and never tests a coefficient for zero; with exact coefficients
            # these would be exact zeros. Our float coefficients carry ~1e-17 noise here.
            shears = [(s, d, cc) for s, d, cc in L if s is not None and abs(cc) >= 1e-12]
            scales = [(d, cc) for s, d, cc in L if s is None]
            if not shears and not scales:
                continue
            if scales and not shears:
                vec = np.ones(r, dtype=np.complex128)
                for d, cc in scales:
                    vec[d] *= cc
                slots.append(('diag', vec))
                continue
            # a shear layer (scales in the same layer are on disjoint coordinates: fold them in)
            # D0 C D1 C D2 C D3 acts on the vector right-to-left: apply D3 first
            for k in (3, 2, 1, 0):
                vec = np.ones(r, dtype=np.complex128)
                for s, d, cc in shears:
                    p = shear_factors(cc)[k]
                    vec[s] = p[0]
                    vec[d] = p[1]
                if k == 3:
                    for d, cc2 in scales:
                        vec[d] *= cc2
                slots.append(('diag', vec))
                if k > 0:
                    slots.append(('C', [(s, d) for s, d, _ in shears]))

    emit_word(tword)
    slots.append(('diag', Dmid))
    emit_word(word)
    slots.append(('diag', Dh))
    return slots


def apply_slots(slots, x):
    y = np.array(x, dtype=np.complex128)
    for kind, data in slots:
        if kind == 'diag':
            y = data * y
        else:
            for s, d in data:
                pair = np.array([y[s], y[d]])
                y[s], y[d] = C @ pair
    return y


def fourier_matrix(u, r):
    return np.array([[u ** (i * j) for j in range(r)] for i in range(r)], dtype=np.complex128)


# ---- fixed slot pattern (step 11c) ------------------------------------------------------
# Every block has the same slot-type pattern  D C D C D C D  (4 monomial, 3 C slots).
# Blocks that need fewer operations are padded with identity slots, so all blocks of all
# words share one pattern (the synchronisation requirement of the proposition).

def _identity_block(r):
    return {'diag': [np.ones(r, dtype=np.complex128) for _ in range(4)], 'pairs': [[], [], []]}


def _pattern_block_from_layer(layer, r):
    """One block for one layer of a word (shear layer or scale-only layer)."""
    blk = _identity_block(r)
    shears = [(s, d, cc) for s, d, cc in layer if s is not None and abs(cc) >= 1e-12]
    scales = [(d, cc) for s, d, cc in layer if s is None]
    for k in range(4):
        vec = blk['diag'][k]
        for s, d, cc in shears:
            p = shear_factors(cc)[3 - k]     # D3 applied first
            vec[s] = p[0]
            vec[d] = p[1]
        if k == 0:
            for d, cc in scales:
                vec[d] *= cc
    if shears:
        blk['pairs'] = [[(s, d) for s, d, _ in shears]] * 3
    return blk


def _diag_block(vec, r):
    blk = _identity_block(r)
    blk['diag'][0] = np.array(vec, dtype=np.complex128)
    return blk


def compile_blocks(u, r):
    """Fixed-pattern block sequence for F_r(u); same action as compile_fourier."""
    H, c = newton_coefficients(u, r)
    # H_r = 0 for a primitive r-th root (1 - u^r = 0), so h_r = 1/H_r is undefined.
    # The width-r word uses only h_0..h_{r-1}; h_r is padding, set to 0 (see exact_check_dft).
    h = [1 / H[j] for j in range(r)] + [0]
    Dh = np.array([H[i] for i in range(r)], dtype=np.complex128)
    Dmid = np.array([c[k] / H[k] for k in range(r)], dtype=np.complex128)
    word = block_word_local(np.array(h, dtype=np.complex128), r)
    tword = transpose_word(word)
    blocks = [_diag_block(Dh, r)]
    for L in tword:
        blocks.append(_pattern_block_from_layer(L, r))
    blocks.append(_diag_block(Dmid, r))
    for L in word:
        blocks.append(_pattern_block_from_layer(L, r))
    blocks.append(_diag_block(Dh, r))
    return blocks


def apply_blocks(blocks, x):
    y = np.array(x, dtype=np.complex128)
    for blk in blocks:
        y = blk['diag'][0] * y
        for k in range(3):
            for s, d in blk['pairs'][k]:
                y[s], y[d] = C @ np.array([y[s], y[d]])
            y = blk['diag'][k + 1] * y
    return y


def pattern_ok(blocks):
    return all(len(b['diag']) == 4 and len(b['pairs']) == 3 for b in blocks)


# ---- zero-free compilation (step 17, local.tex eq. loc:nonzero-split) -------------------
# Every shear with coefficient mu is replaced by two shears with coefficients t1, t2 that are
# nonzero for EVERY mu (including mu = 0):  kappa = 1 + |mu|^2,  t1 = kappa,  t2 = mu - kappa.
# Then t1 + t2 = mu and both are nonzero (1 + |mu|^2 > |mu| makes t2 != 0). Lower shears commute,
# so the two blocks compose to the desired shear. No test for zero is performed anywhere.

def split_shear(mu):
    mu = complex(mu)
    kappa = 1 + abs(mu) ** 2
    return kappa, mu - kappa


def zero_free_blocks(u, r, *, strict=True):
    """Zero-free blocks; strict topology by default, legacy pruning only with strict=False."""
    H, c = newton_coefficients(u, r)
    # H_r = 0 for a primitive r-th root (1 - u^r = 0), so h_r = 1/H_r is undefined.
    # The width-r word uses only h_0..h_{r-1}; h_r is padding, set to 0 (see exact_check_dft).
    h = [1 / H[j] for j in range(r)] + [0]
    Dh = np.array([H[i] for i in range(r)], dtype=np.complex128)
    Dmid = np.array([c[k] / H[k] for k in range(r)], dtype=np.complex128)
    with zero_multiplier_mode(strict):
        word = block_word_local(np.array(h, dtype=np.complex128), r)
    tword = transpose_word(word)

    def diag_blk(vec):
        blk = _identity_block(r)
        blk['diag'][0] = np.array(vec, dtype=np.complex128)
        return blk

    def layer_blocks(layer):
        shears = [(s, d, cc) for s, d, cc in layer if s is not None]
        scales = [(d, cc) for s, d, cc in layer if s is None]
        out = []
        for part in (0, 1):
            blk = _identity_block(r)
            for k in range(4):
                vec = blk['diag'][k]
                for s, d, cc in shears:
                    t = split_shear(cc)[part]
                    p = shear_factors(t)[3 - k]
                    vec[s] = p[0]
                    vec[d] = p[1]
                if k == 0 and part == 0:
                    for d, cc2 in scales:
                        vec[d] = vec[d] * cc2
            if shears:
                blk['pairs'] = [[(s, d) for s, d, _ in shears]] * 3
            out.append(blk)
        return out

    blocks = [diag_blk(Dh)]
    for L in tword:
        blocks += layer_blocks(L)
    blocks.append(diag_blk(Dmid))
    for L in word:
        blocks += layer_blocks(L)
    blocks.append(diag_blk(Dh))
    return blocks
