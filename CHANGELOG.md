# Changelog

Revision history of the study. Every retraction and correction below concerns
our own implementation, experiments or text. None of them revealed an error in
the preprint. The authoritative account is section 1.5 of
[`paper/article-ua.txt`](paper/article-ua.txt).

## 1.0.0 — 2026-10-14 (public release of manuscript v6.3)

- First public release: library `allength/`, 123 unit tests, experiment and
  validation scripts `paper/scripts/`, raw logs and aggregated data
  `paper/data/`, the reproduction package `release/`, and the frozen evidence of
  the completed core run `release/evidence/full-20261013-r3/` (135/135 steps).
- The paper in LaTeX: English primary edition `paper/article-en.pdf` and the
  parallel Ukrainian edition `paper/article-uk.pdf`; tables are generated from
  the validated manuscript by `paper/tex/gen_tables.py`.
- Licensed under Apache License 2.0.

## v6.3 — 2026-10-14

- The core run `full-20261013-r3` completed all 135 steps. Integrity of the
  captured sources, logs, outputs, environment and the raw → CSV aggregation
  (no substitution of archived results) was checked independently of the run
  itself; the current table validator matched the fresh data against the text.
- Historical timings kept; the repeated timings are reported separately
  (Table 8a).
- Added a separate streamed F_p check at r = 1024 (two fixed vectors in two
  fields, 0/4096 mismatches; a necessary condition only).
- Clarified that the Lean formalization matches the preprint's main goal via a
  different network, the scale of the local widths, and the strength of each
  kind of check.
- Wording fix: the convergence gain of the full DFT chain is 99.7–100.2 orders
  of magnitude; the earlier range 99.7–100.3 mixed in the value 100.3, which
  belongs to the C-slot path at r = 256.

## v6.2

Two bugs were found in our own phase-precision control script
(`paper/scripts/phase_precision.py`, Table 2d of v6.1):

1. a coefficient-times-coordinate product and the following addition were
   rounded as a single operation, whereas ordinary execution of the word rounds
   them separately;
2. stage T received the input D_H·x instead of its actual input
   Dmid·Tᵀ·D_H·x.

Table 2d was recomputed. The v6.1 conclusion that almost all error originates in
the chunk phase and is merely carried afterwards was withdrawn; Table 2d and its
conclusions are now limited to the confirmed sensitivity to phase precision.
Aggregated CSVs were completed with existing results and short exact checks.
The strict zero-free API no longer depends on the initial global flag.

## v6.1

Self-check and pre-publication audit of the whole study.

- Added: growth breakdown of intermediate magnitudes by recursion level and
  phase (Table 2c); a line-by-line comparison of the Lean definitions with the
  preprint's model (section 3.1, `release/lean/MODEL_AUDIT.md`); a convergence
  check of the C-slots at r = 256; the table validator
  `paper/scripts/verify_tables.py`; `--data-dir` for `make_figures.py`.
- Corrected: the Clausen-function formula is stated as the leading asymptotic
  term, not an exact value; Table 2c is described as a growth breakdown, not an
  error breakdown, with a high-precision phase control (Table 2d); convergence
  is stated as "≈100 orders with deviations of a few tenths".
- Fixed a `ZeroDivisionError` in five compiler functions on exactly
  representable roots (h_r = 0); isolated precision in tests; fixed a table
  rendering defect in the PDF and added an arity check to the builder; replaced
  the system lock file with a portable one; extended `references.bib` to 11
  entries; clarified the F_p homomorphism, the conjugation automorphism, the
  set of checked widths and the status of obsolete files.

## v6

- Added exact checks in cyclotomic fields Q(ζ_N) and finite fields F_p, odd
  widths, a second precision for all widths, C-slots up to r = 256 with
  conjugation through a DAG with inverted roots, stage-wise error and
  conditioning analysis, the roots of h(z), a full topology audit and a static
  audit of value-dependent branches, Table 3 in strict mode, time and memory
  measurements, a literature survey and the reproduction package.
- The statement "the implemented operator equals F_r(u)" is now reserved for
  exact checks; for multiprecision results we write "agrees with exact
  equality".

## v5 — value-dependent pruning replaced by the strict construction

Version v4 measured our own variant of the compiler, which deviated from the
preprint: the convolution DAG skipped a pointwise multiplier whenever its
coefficient was exactly zero (`conv_dag.py`, condition `abs(coef) == 0`). The
operator is unchanged by this, but the decision depends on a zero test of a
complex coefficient that the preprint's construction does not use, and it makes
the topology depend on coefficient values (at r = 64, 2024 of 12 066 pointwise
multipliers were skipped for one input and 2002 for another). The preprint
requires the topology to depend only on r and keeps all three outer products
even when some coefficients vanish.

From v5 on, the main object is the strict construction
(`conv_dag.KEEP_ZERO_MULTIPLIERS = True`), and the following v4 claims were
retracted:

- the depth crossover between r = 320 and 336 (in strict mode the adaptive rule
  has no greater depth at all 12 measured widths and strictly smaller from
  r = 128, Table 5);
- a seed-to-seed depth spread of order 10⁻⁴ as a property of the construction
  (absent in strict mode, Table 4);
- the claim that Table 1 of v4 confirms correctness of the whole chain (it only
  checked T·x for random h);
- the wording "error only at the level of arithmetic precision" (tens of digits
  are lost);
- the comparison "hundreds of times more expensive than an FFT", which divided
  complex gates by real flops (Table 7);
- the analysis in §7.1 that accounted only for the factor (log n)^{−δ}.

## v4 — 2026-10-12

- Internal draft; superseded. Its depth and seed-spread results were artefacts
  of value-dependent zero-multiplier pruning (see v5).

## Before v4

- Retracted our results obtained on the degenerate input h = ones.

## Initial public Git package — 2026-10-14

- Include the current English LaTeX/PDF as a working draft.
- State the six concrete contributions and bound the interpretation of phase
  diagnostics; no new DFT algorithm or mathematical priority is claimed.
- Update the Ukrainian numerical validation reference editorially, preserving
  every numerical table; keep executed evidence byte-for-byte.
- Preserve bytes across Git clones to keep checksum validation portable.
