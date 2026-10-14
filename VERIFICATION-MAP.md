# Verification map: "An explicit power saving for the exact discrete Fourier transform"

> **Archived (pre-v4 state).** The current statement-by-statement status is Table 0 and
> sections 3-6 of `paper/article-ua.txt` (v6.3); reproduction is `release/README_REPRODUCE.md`.
> This map is kept only for history and may be out of date.

Each row: paper statement -> lab module -> test file -> what is checked -> what is NOT checked.
All tests run with `.venv/bin/python -m unittest tests.<name>`. Float64 unless noted (mp = mpmath).

| Paper statement | Module | Test | Checked | Not checked |
|---|---|---|---|---|
| Classical arbitrary-length DFT (CRT, Bluestein) | allength/dft_all.py | test_allength | DFT n=1..200, large n vs np.fft | paper's own fast compiler |
| Newton reduction lemma loc:newton-reduction | allength/newton_local.py | test_newton_local | exact identity (80 digits), closed form, Toeplitz structure, float conditioning | - |
| Replay with borrowed coords lemma loc:replay | allength/replay.py | test_replay | exact on integers, arbitrary/correlated z, mutations | depth constant |
| Convolution DAG eq. loc:convolution-bounds | allength/conv_dag.py | test_conv_dag | exactness, gates ~ v log v, depth ~ log v, fan-out <= 2 | - |
| Toeplitz block split, displacement, two-convolution apply | allength/toeplitz_disp.py | test_toeplitz_disp | all algebraic steps | exact width (see exact_width) |
| Exact-width block Toeplitz word | allength/exact_width.py, mp_exact.py | test_exact_width, test_mp_exact | r<=512 at 60-320 digits (block-local; r=512 err 1.6e-212 at 320 digits) | round count (see rounds) |
| Round count O(log^4 r) (loc:toeplitz) | allength/rounds.py, depth_recurrence.py | (manual runs) | depth recurrence, plateau d/log^4 r over r=2^8..2^16 | proof of constant |
| Uniform local compiler loc:compiler, Fourier layers loc:fourier | allength/compiler.py, mp_compiler.py | test_compiler, test_mp_compiler, test_zero_free | closed-form shears, fixed slot pattern, all primitive roots r<=12 (float), 60 digits (mp), zero-free split | exact cyclotomic arithmetic |
| All-length assembly | allength/assembly.py | test_assembly | DFT n=2..120 from compiled kernels | Bluestein inner FFT not compiled |
| Sector packing (sync. lemma sector-address) | allength/packing.py | test_packing | bijection, incremental = brute, start formula, mutation | operation cost |
| Prefix traversal eq. prefix-nodes | (inline) | - | 20000 random radix tuples | - |
| Tensor Fourier proposition tensor-fourier (correctness) | allength/tensor_sync.py | test_tensor_sync | kron of per-axis words | cost bound |
| Synthesis certificate, second clause (I+E12) | allength/compiler.py | test_three_call | 3-call word for I+E12 | first clause (network) |
| Printer radix selection, split identity (synthesis) | allength/printer.py | test_printer | selection rule with B-bit truncation; index identity | the bit machine itself (not in paper) |
| Critical exponent theorem net:tensor-bound | allength/batching.py | test_batching | recurrence unrolls to k^theta | concrete W_*, S of the network |
| Network counts (wires, invocations, neighbours) | allength/network_small.py | test_network_small | explicit enumeration h=5,6,7; formulas at h=100 | production-size network |
| Lemma net:scalar (RG+JV=Id, invocation, 3 stages) | allength/network_small.py | test_network_small | exact rationals h=5,6,7; full grid T^3 for 3 stages | full row schedule in the grid |
| Lemma net:residuals (s, Delta) | (arithmetic) | test_network_small | exact s and Delta = 6871402692000000 | the residual graph |
| Subspace decomposition (net:residuals proof) | allength/subspace_decomp.py | test_subspace_decomp | n=3..6 brute force | - |
| Mod-4 weight identity (net:finite-interface proof) | allength/frame_mod4.py | test_frame_mod4 | m=1..6 brute force | the frame construction |
| Edge kernel scalar and Walsh identities (net:edge) | allength/edge_kernels.py | test_edge_kernels | i^{[s]}=a+b(-1)^s, C^2=R, C^{-1}=RC, P_z = H(aI+bT_z)H for m=2..5 | the paper's edge operator on frames |
| Indexing data movement (net:indexing) | allength/indexing.py | test_indexing | table = direct on 300 instances | O(2^k) time |
| Lean exponent constants (Goal.lean) | (numeric) | - | alpha < theta < 1, alpha < decimal < 1 | Lean compilation |
| Lean: C matrix and phase relation | (static read) | - | ca, cb, littleC, littleH relation match | - |

## Not verified (blocking)
- Lean/Comparator build of the formal proof: no toolchain on this machine (needs elan, Lean v4.34.1, mathlib; a multi-GB download that requires user permission).
- Production-size network (h=100, 10^6 wires, 3 stages with full invocations): infeasible in Python.
- Synthesis printer as an actual bit machine: the paper only asserts existence.
- Operation-count and depth bounds as stated in the paper: measured only on small sizes; the paper's constants are not reproduced.
- Exact arithmetic (cyclotomic field): zero structure is handled by the zero-free split, but coefficients remain float64 (or 60-digit mp).

## Discrepancy found
The Lean formalisation (lean/OAI/Computability/FourierTransform/Main.lean docstring) uses a smaller network (4060 triples = C(30,3)) and does not formalise the synthesis appendix. Its stated constants (paperDelta, paperMultiplier) are consistent with the paper's h=100 counts.
