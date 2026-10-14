"""Wall time and peak memory of OUR Python implementation (article v6, table 8).

Usage:  python timing.py R
Measures, in one fresh process per R (strict mode, float64, adaptive rule):
  build   : time to prepare the word (reciprocal series, chunk DAGs, replay layers, all scalars);
  apply   : time to apply the stored word to one vector (pure Python loop);
  rss     : peak resident memory of the process (ru_maxrss);
  numpy   : median time of numpy.fft.fft on length R and of a dense lower-triangular
            Toeplitz mat-vec (numpy) on the same machine, for scale.
These are properties of this implementation and machine, not of the algorithm.
"""
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np  # noqa: E402

from allength import conv_dag as CD  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True
from allength import rounds as RD  # noqa: E402

R = int(sys.argv[1])
rng = np.random.default_rng(R)
h = rng.standard_normal(R + 1) + 1j * rng.standard_normal(R + 1)
h[0] = abs(h[0]) + 1
x = rng.standard_normal(R) + 1j * rng.standard_normal(R)

t = time.perf_counter()
word = RD.block_word_local(h, R)
t_build = time.perf_counter() - t
t = time.perf_counter()
RD.execute_sequential(x, word)
t_apply = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024   # GB (ru_maxrss in KB)


def med(f, n=200):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


Tm = np.zeros((R, R), dtype=np.complex128)
for i in range(R):
    Tm[i, :i + 1] = h[i::-1][:i + 1]
t_fft = med(lambda: np.fft.fft(x))
t_mv = med(lambda: Tm @ x)
gates = sum(len(L) for L in word)
print(f"r={R} gates={gates} build={t_build:.2f}s apply={t_apply:.2f}s peak_rss={rss:.2f}GB "
      f"numpy_fft={t_fft * 1e6:.1f}us numpy_dense_toeplitz={t_mv * 1e6:.1f}us", flush=True)
