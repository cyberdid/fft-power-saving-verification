"""Phase-precision interventions for a strict Toeplitz word.

Coefficients are prepared at reference precision and held fixed across interventions.
This experiment isolates rounding during data operations; coefficient-preparation error
is a separate source. A high-precision phase is a reference computation, not exact algebra.
Usage: python phase_precision.py R DPS [--stage tword|word] [--guard-digits 150]
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp
from allength import conv_dag as CD
from allength.compiler import transpose_word
from paper.scripts.causal_levels import tagged_block_word


def round_to(z, dps):
    with mp.workdps(dps):
        return mp.mpc(z.real, z.imag)


def replay(word, tags, y, exact_phases, dps):
    """Replay reference-coefficient layers, rounding the selected data operations."""
    if len(word) != len(tags):
        raise ValueError("every layer must have a phase tag")
    y = list(y)
    for layer, (_, phase) in zip(word, tags):
        for src, dst, coefficient in layer:
            def operation():
                if src is None:
                    return coefficient * y[dst]
                # A shear is a multiplication followed by an addition. Both
                # operations round in the working context; this is not an FMA.
                return y[dst] + coefficient * y[src]
            if phase in exact_phases:
                y[dst] = operation()
            else:
                with mp.workdps(dps):
                    y[dst] = operation()
    return y


def norm(vector):
    return mp.sqrt(sum(abs(x)**2 for x in vector))


def build_experiment(r, dps, stage, guard_digits=150):
    """Return the stage word, tags and input at caller's reference precision."""
    if stage not in ("tword", "word") or r < 1 or dps < 2 or guard_digits < 50:
        raise ValueError("invalid experiment parameters")
    u = mp.exp(2j * mp.pi / r)
    H = [mp.mpc(1)]
    for k in range(1, r):
        H.append(H[-1] * (1-u**k))
    h = [1/v for v in H] + [mp.mpc(0)]
    previous = CD.KEEP_ZERO_MULTIPLIERS
    CD.KEEP_ZERO_MULTIPLIERS = True
    try:
        word, tags = tagged_block_word(h, r)
    finally:
        CD.KEEP_ZERO_MULTIPLIERS = previous
    rng = random.Random(r)
    x = [mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(r)]
    yin = [H[k]*x[k] for k in range(r)]
    if stage == "tword":
        word, tags = transpose_word(word), list(reversed(tags))
    else:
        # The actual T input is Dmid*T^T*D_H*x, not D_H*x.
        transposed = [sum(h[j-i] * yin[j] for j in range(i, r))
                      for i in range(r)]
        yin = [(-1)**k*u**(k*(k-1)//2)/H[k]*transposed[k] for k in range(r)]
    return word, tags, yin, h[:r], H, x


def run_experiment(r, dps, stage="tword", guard_digits=150):
    with mp.workdps(dps+guard_digits):
        word, tags, yin, h, H, x = build_experiment(r, dps, stage, guard_digits)
        phases = {phase for _, phase in tags}
        reference = replay(word, tags, yin, phases, dps)
        def loss(high_phases):
            output = replay(word, tags, yin, high_phases, dps)
            error = norm([a-b for a,b in zip(output, reference)])/norm(reference)
            return float(dps+mp.log10(error)) if error else -math.inf
        baseline = loss(set())
        interventions = {phase: loss({phase}) for phase in sorted(phases)}
        complement = loss(phases-{"chunk"}) if "chunk" in phases else None
        return {"r":r,"dps":dps,"stage":stage,"reference_dps":dps+guard_digits,
                "phases":sorted(phases),"baseline_lost":baseline,
                "high_precision_lost":interventions,"chunk_only_low_lost":complement}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r", type=int)
    parser.add_argument("dps", type=int)
    parser.add_argument("--stage", choices=("tword","word"), default="tword")
    parser.add_argument("--guard-digits", type=int, default=150)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(args.r,args.dps,args.stage,args.guard_digits)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        def safe(value):
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            return None if isinstance(value, float) and not math.isfinite(value) else value
        args.output_json.write_text(json.dumps(safe(result), indent=2, allow_nan=False)+"\n")
    print(f"r={args.r} dps={args.dps} stage={args.stage} phases={result['phases']} reference={result['reference_dps']}")
    base = result["baseline_lost"]
    print(f"  baseline (all phases at DPS):           digits lost = {base:6.1f}")
    for phase, lost in result["high_precision_lost"].items():
        removed = base-lost if math.isfinite(base) and math.isfinite(lost) else math.nan
        print(f"  {phase:<7} high precision, rest at DPS: digits lost = {lost:6.1f} (removes {removed:5.1f} digits)")
    if result["chunk_only_low_lost"] is not None:
        print(f"  all high precision EXCEPT chunk:        digits lost = {result['chunk_only_low_lost']:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
