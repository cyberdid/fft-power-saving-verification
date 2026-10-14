"""Step 27: column data movement of lemma net:indexing (network.tex).

Input: f columns of m bits (address k = m*f + r; r spectator bits). Operations:
  1. apply a fixed invertible binary map M (m x m over F_2) to each column;
  2. translate each column by a fixed vector c_j;
  3. collect one chosen output bit from every column into a contiguous f-bit field.
Table-driven implementation (one table per column digit, built once) versus the direct definition.
"""

import itertools
import random


def mat_apply(M, x):
    """M: list of row bitmasks (m rows); x: integer bitmask of m bits."""
    y = 0
    for i, row in enumerate(M):
        if bin(row & x).count('1') % 2:
            y |= 1 << i
    return y


def direct(M, c, sel, cols):
    """Direct definition: out bit j = bit sel[j] of (M*x_j + c_j) for each column j."""
    out = 0
    for j, x in enumerate(cols):
        y = mat_apply(M, x) ^ c[j]
        out |= ((y >> sel[j]) & 1) << j
    return out


def tabled(M, c, sel, cols, m):
    """Same result via a table over all column digits, built once per fixed (M, sel)."""
    table = [mat_apply(M, x) for x in range(1 << m)]
    out = 0
    for j, x in enumerate(cols):
        y = table[x] ^ c[j]
        out |= ((y >> sel[j]) & 1) << j
    return out


def invertible(M, m):
    vals = {mat_apply(M, x) for x in range(1 << m)}
    return len(vals) == (1 << m)


def check(seed=0, trials=300):
    rng = random.Random(seed)
    done = 0
    for _ in range(trials):
        m = rng.randint(1, 4)
        f = rng.randint(1, 4)
        while True:
            M = [rng.randrange(1 << m) for _ in range(m)]
            if invertible(M, m):
                break
        c = [rng.randrange(1 << m) for _ in range(f)]
        sel = [rng.randrange(m) for _ in range(f)]
        cols = [rng.randrange(1 << m) for _ in range(f)]
        assert direct(M, c, sel, cols) == tabled(M, c, sel, cols, m)
        done += 1
    return done
