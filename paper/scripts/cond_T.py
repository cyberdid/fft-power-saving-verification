"""log10 cond_2 of the Fourier-case Toeplitz factor T (T_ij = 1/H_{i-j}) and of D_H.
Usage: python cond_T.py R DPS   (mpmath SVD)"""
import sys
import mpmath as mp
R, DPS = int(sys.argv[1]), int(sys.argv[2])
mp.mp.dps = DPS
u = mp.exp(2j * mp.pi / R)
H = [mp.mpc(1)]
for j in range(1, R):
    H.append(H[-1] * (1 - u ** j))
T = mp.matrix(R, R)
for i in range(R):
    for j in range(i + 1):
        T[i, j] = 1 / H[i - j]
sv = mp.svd_c(T, compute_uv=False)
s = [abs(sv[i]) for i in range(R)]
print(f"r={R}: log10 cond2(T)={float(mp.log10(max(s)/min(s))):.2f} log10 smax={float(mp.log10(max(s))):.2f} "
      f"log10 smin={float(mp.log10(min(s))):.2f} log10 cond2(Dh)={float(mp.log10(max(abs(v) for v in H)/min(abs(v) for v in H))):.2f}", flush=True)
