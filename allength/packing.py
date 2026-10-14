"""Step 14: sector packing (synchronization.tex, lemma sector-address, eqs. sector-start/updates).

Axes j = 0..l-1 with radices r_j. On each axis the digits are partitioned into ordered blocks of
size q in {1,2}. A sector chooses one block per axis; its width is Q = prod q_j. Sectors are ordered
lexicographically by their block choices; inside a sector, digits are ordered by the local positions
t_j (first axis most significant). The packed index of a coordinate is S + o, where S is the sector
start and o the offset inside the sector.
"""

import itertools
import random


def random_partition(rng, r):
    """Random ordered partition of range(r) into blocks of size 1 or 2 (lists of digits)."""
    digits = list(range(r))
    rng.shuffle(digits)
    blocks = []
    while digits:
        if len(digits) >= 2 and rng.random() < 0.5:
            blocks.append([digits.pop(), digits.pop()])
        else:
            blocks.append([digits.pop()])
    return blocks


def local_tables(blocks):
    """For each axis: per digit a -> (p, q, t) with p = coordinates in preceding blocks,
    q = size of its block, t = position inside the block."""
    tab = {}
    p = 0
    for blk in blocks:
        for t, a in enumerate(blk):
            tab[a] = (p, len(blk), t)
        p += len(blk)
    return tab


def brute_packing(radices, partitions):
    """Brute force: enumerate sectors lexicographically, assign packed indices to all coordinates."""
    l = len(radices)
    tabs = [local_tables(partitions[j]) for j in range(l)]
    sector_choices = [range(len(partitions[j])) for j in range(l)]
    packed = {}
    base = 0
    for choice in itertools.product(*sector_choices):
        blocks = [partitions[j][choice[j]] for j in range(l)]
        Q = 1
        for blk in blocks:
            Q *= len(blk)
        # enumerate offsets: t-digits, first axis most significant
        for tt in itertools.product(*[range(len(blk)) for blk in blocks]):
            o = 0
            for j in range(l):
                o = o * len(blocks[j]) + tt[j]
            digits = tuple(blocks[j][tt[j]] for j in range(l))
            addr = 0
            for j in range(l):
                addr = addr * radices[j] + digits[j]
            packed[addr] = base + o
        base += Q
    return packed, base


def formula_start(radices, partitions, choice, tabs_p_q=None):
    """S = sum_j (prod_{i<j} q_i) * p_j * tau_j for the sector with the given block choice."""
    l = len(radices)
    S = 0
    for j in range(l):
        blk = partitions[j][choice[j]]
        p_j = sum(len(partitions[j][k]) for k in range(choice[j]))
        tau_j = 1
        for i in range(j + 1, l):
            tau_j *= radices[i]
        prod_q = 1
        for i in range(j):
            prod_q *= len(partitions[i][choice[i]])
        S += prod_q * p_j * tau_j
    return S


def incremental_packing(radices, partitions):
    """Traversal with the four states (S, Q, o, I) of eq. sector-updates, per block choice path."""
    l = len(radices)
    tabs = [local_tables(partitions[j]) for j in range(l)]
    out = {}

    def recurse(j, S, Q, o, I, path):
        if j == l:
            out[I] = S + o
            return
        # every digit of this axis belongs to exactly one block; traverse digits through blocks
        for a in range(radices[j]):
            p, q, t = tabs[j][a]
            # S and Q updated per digit prefix; sector choice is implied by the digit's block
            recurse(j + 1, S + Q * p * _tau(radices, j), Q * q, o * q + t, I * radices[j] + a,
                    path)

    def _tau(radices, j):
        t = 1
        for i in range(j + 1, l):
            t *= radices[i]
        return t

    recurse(0, 0, 1, 0, 0, ())
    return out


def check(seed, trials=300):
    rng = random.Random(seed)
    for trial in range(trials):
        l = rng.randint(1, 3)
        radices = [rng.randint(2, 4) for _ in range(l)]
        partitions = [random_partition(rng, r) for r in radices]
        packed, total = brute_packing(radices, partitions)
        R = 1
        for r in radices:
            R *= r
        assert total == R, "packed indices must be a bijection onto 0..R-1"
        assert sorted(packed.values()) == list(range(R))
        inc = incremental_packing(radices, partitions)
        assert inc == packed, f"incremental traversal disagrees, radices={radices}"
        for choice in itertools.product(*[range(len(partitions[j])) for j in range(l)]):
            S = formula_start(radices, partitions, choice)
            # brute sector start for this choice
            blocks = [partitions[j][choice[j]] for j in range(l)]
            digits0 = tuple(blk[0] for blk in blocks)
            addr0 = 0
            for j in range(l):
                addr0 = addr0 * radices[j] + digits0[j]
            assert packed[addr0] - 0 >= S, "sector start formula inconsistent"
            # the first coordinate of the sector sits exactly at S (offset 0)
            assert packed[addr0] == S, f"start mismatch radices={radices} choice={choice}"
    return trials
