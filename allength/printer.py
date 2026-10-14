"""Step 19: printer radix selection and the split index identity (synthesis.tex, theorem syn:printer).

select_radix: for input length n, B = floor(log2 n) bit steps of the printer have produced records
with radices m_t (strictly increasing); select the LAST record with 2 <= m_t <= sqrt(B); if none,
the ordinary radix 2.
split identity: for input index i = b*j_a + j_b and output index k = k_a + a*k_b with
0<=j_a,k_a<a and 0<=j_b,k_b<b:  i*k == b*j_a*k_a + a*j_b*k_b + j_b*k_a  (mod a*b).
"""

import math


def select_radix(record_radices, n):
    """record_radices: the m_t emitted so far (in order). Returns the selected radix."""
    if n < 2:
        raise ValueError("n >= 2")
    B = int(math.floor(math.log2(n)))
    limit = math.isqrt(B)
    emitted = record_radices[:B]          # at most one record per bit step, so only B bits seen
    chosen = 2
    for m in emitted:
        if 2 <= m <= limit:
            chosen = m                    # keep the last one satisfying the bound
    return chosen


def split_identity_holds(a, b):
    for ja in range(a):
        for jb in range(b):
            for ka in range(a):
                for kb in range(b):
                    i = b * ja + jb
                    k = ka + a * kb
                    if (i * k - (b * ja * ka + a * jb * kb + jb * ka)) % (a * b) != 0:
                        return False
    return True
