#!/usr/bin/env python3
"""Streaming exact F_p check of the strict DFT word F_r(u) = Dh T Dmid T^T Dh.

Read-only use of the project selected by --project (imported via sys.path; nothing
there is modified, no bytecode is written). The portable default searches the script's
ancestors, then the working directory, for allength/; it works from paper/scripts/.

Same inputs as paper/scripts/exact_check_dft.py (prime mode, random vectors):
  N = lcm(R, 2^ceil(log2(8R))), primes p = PrimeField(N, index=k) (smallest p = 1 mod N above
  2^61), vectors drawn by random.Random(R).randrange(p) (fresh generator for every prime),
  h = [1/H_j for j < R] + [0]  (h_R padding = 0, H_R is never inverted).

Streaming: the word of g_block_word_local is regenerated layer by layer, chunk-pair DAG by
chunk-pair DAG, and every layer is applied to all vectors immediately and then dropped.
  * forward generator  = exact replica of g_block_word_local's recursion (program order);
  * reverse generator  = the same recursion mirrored: chunk pairs of a block in reverse order,
    each pair's replay layers reversed, then the right half, then the left half; direct
    (b = 0) blocks and single scales are reversed explicitly.  The transpose word
    (allength.compiler.transpose_word) is then obtained on the fly by swapping shear ends of
    each layer (src, dst, c) -> (dst, src, c); scales (None, d, c) are unchanged; the order of
    gates inside a layer is kept, exactly as transpose_word does.
  Peak memory: h, g (length R+1), the vectors, and ONE chunk-pair DAG with its replay layers.
  Each pass rebuilds the pair DAGs (the transpose pass and the forward pass both build them),
  i.e. we trade 2x DAG-construction time for O(one DAG) memory instead of O(whole word).

Self-checks recorded in the result JSON:
  * sha256 of the applied gate stream (layer index in application order, src, dst, coef mod p)
    for the transpose pass and the forward pass (compared against the materialised builder with
    --materialise-check);
  * a positional polynomial hash sum_k d(L_k) B^k mod (2^127-1) of the FORWARD layers L_k:
    accumulated with increasing powers in the forward pass and by Horner's rule in the reverse
    pass; a separate Horner hash always follows the ACTUAL emitted input-layer order,
    including negative controls. Equality is a fingerprint of reversed order, subject to
    hash collisions; it does not check the subsequent swapping of shear ends.
  * execution-time SHA-256 snapshots of this script and imported allength source modules,
    before and after the checks (on-disk source fingerprints, not bytecode attestation).

Usage:
  stream_exact_dft.py R [--primes K] [--nvec N | --basis] [--materialise-check]
                        [--corrupt] [--json PATH] [--project PATH]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

CHECKER_VERSION = "codex-audit-v2"


def default_project():
    """Find a project relative to this script, including a future paper/scripts copy."""
    for candidate in (*Path(__file__).resolve().parents, Path.cwd()):
        if (candidate / "allength" / "generic_word.py").is_file():
            return str(candidate)
    return str(Path(__file__).resolve().parents[2])


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("R", type=int)
    ap.add_argument("--project", default=default_project(),
                    help="project containing allength/ (default: script ancestor, then cwd)")
    ap.add_argument("--primes", type=int, default=2)
    ap.add_argument("--first-prime", type=int, default=0)
    ap.add_argument("--nvec", type=int, default=2)
    ap.add_argument("--basis", action="store_true")
    ap.add_argument("--materialise-check", action="store_true")
    ap.add_argument("--corrupt", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--bad-transpose", choices=["noswap", "noreverse"], default=None,
                    help="negative control only: deliberately wrong transpose handling")
    return ap


# Select the project before importing any project code; --help also works independently.
_PARSER = argument_parser()
_ARGS = _PARSER.parse_args() if __name__ == "__main__" else None
PROJECT = str(Path(_ARGS.project if _ARGS is not None else default_project()).expanduser().resolve())
if not (Path(PROJECT) / "allength" / "generic_word.py").is_file():
    _PARSER.error(f"no allength/generic_word.py in {PROJECT}; pass --project PATH")
sys.path.insert(0, PROJECT)

from allength import conv_dag as CD  # noqa: E402

CD.KEEP_ZERO_MULTIPLIERS = True          # strict mode, always

from allength.chunk_dag import sum_dags  # noqa: E402
from allength.exact_field import GF, PrimeField, lcm  # noqa: E402
from allength.generic_word import g_chunk_parts, g_outer_dag, g_reciprocal  # noqa: E402
from allength.rounds import admissible_chunk, replay_layers  # noqa: E402

MERSENNE = (1 << 127) - 1
BAD_TRANSPOSE = None
POLY_B = 0x5DEECE66D1F2E3C4B5A69788ABCDEF1  # fixed base for the positional layer hash
GATE = struct.Struct("<qqqQ")              # (layer index, src or -1, dst, coef mod p)
RAW = struct.Struct("<qqQ")                # (src or -1, dst, coef mod p) for the layer digest


def source_snapshot():
    """Fingerprint source files on disk now; include lazily imported modules at the end."""
    def entry(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    modules = []
    for name, module in sorted(list(sys.modules.items())):
        if name == "allength" or name.startswith("allength."):
            path = getattr(module, "__file__", None)
            if path is not None:
                item = entry(path)
                if not Path(item["path"]).is_relative_to(Path(PROJECT)):
                    raise RuntimeError(f"{name} imported outside selected project: {item['path']}")
                modules.append({"name": name, **item})
    return {"captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": entry(__file__), "project_modules": modules}


def sources_unchanged(start, end):
    """Compare all sources present at startup; the end snapshot may include lazy imports."""
    old = {x["path"]: x["sha256"] for x in [start["script"], *start["project_modules"]]}
    new = {x["path"]: x["sha256"] for x in [end["script"], *end["project_modules"]]}
    return all(new.get(path) == digest for path, digest in old.items())


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cur_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


# --------------------------------------------------------------------------- inputs
def fourier(F, R):
    """Identical to exact_check_dft.fourier."""
    H = [F.one()]
    for j in range(1, R + 1):
        H.append(H[-1] * (F.one() - F.root(R, j)))
    cc = [F.root(R, k * (k - 1) // 2) * (1 if k % 2 == 0 else -1) for k in range(R)]
    h = [F.one() / H[j] for j in range(R)] + [F.zero()]
    return H[:R], [cc[k] / H[k] for k in range(R)], h


# --------------------------------------------------------------------------- generators
class WordStream:
    """Forward / reverse layer generators of g_block_word_local(F, h, R) without materialising."""

    def __init__(self, F, h, R):
        self.F, self.h, self.R = F, h, R
        self.g = g_reciprocal(F, h, R + 1)        # same call as g_block_word_local
        self._b = {}
        self.pairs_built = 0
        self.max_pair_gates = 0
        self.max_pair_layers = 0

    def chunk(self, v):
        if v not in self._b:
            self._b[v] = admissible_chunk(v)
        return self._b[v]

    def pair(self, lo, hi, p, b, i0, j0):
        """One chunk pair, exactly as in g_block_word_local: (to_global, replay layers)."""
        F, h, g = self.F, self.h, self.g
        I = list(range(i0, min(i0 + b, hi)))
        J = list(range(j0, min(j0 + b, lo + p)))
        parts = g_chunk_parts(F, h, g, p, i0 - lo, len(I), j0 - lo, len(J))
        dag = sum_dags([g_outer_dag(F, vv, ww, len(J)) for vv, ww in parts])
        used = set(I) | set(J)
        borrowed = [q for q in range(lo, hi) if q not in used]
        assert len(borrowed) >= dag.g, "pair does not fit its block"
        Z = borrowed[:dag.g]
        tg = list(J) + list(I) + list(Z)
        layers = replay_layers(dag)
        self.pairs_built += 1
        self.max_pair_gates = max(self.max_pair_gates, dag.g)
        self.max_pair_layers = max(self.max_pair_layers, len(layers))
        return tg, layers

    @staticmethod
    def _map(tg, L):
        return [(tg[s] if s is not None else None, tg[d], c) for s, d, c in L]

    def forward(self, lo=0, hi=None):
        if hi is None:
            hi = self.R
        h = self.h
        v = hi - lo
        if v == 1:
            yield [(None, lo, h[0])]
            return
        b = self.chunk(v)
        if b == 0:
            for i in range(v - 1, -1, -1):
                yield [(None, lo + i, h[0])]
                for j in range(i):
                    yield [(lo + j, lo + i, h[i - j])]
            return
        p = v // 2
        yield from self.forward(lo, lo + p)
        yield from self.forward(lo + p, hi)
        for i0 in range(lo + p, hi, b):
            for j0 in range(lo, lo + p, b):
                tg, layers = self.pair(lo, hi, p, b, i0, j0)
                for L in layers:
                    yield self._map(tg, L)
                del tg, layers

    def reverse(self, lo=0, hi=None):
        """Layers of the same word in exactly reversed order (contents not yet transposed)."""
        if hi is None:
            hi = self.R
        h = self.h
        v = hi - lo
        if v == 1:
            yield [(None, lo, h[0])]
            return
        b = self.chunk(v)
        if b == 0:
            for i in range(0, v):
                for j in range(i - 1, -1, -1):
                    yield [(lo + j, lo + i, h[i - j])]
                yield [(None, lo + i, h[0])]
            return
        p = v // 2
        for i0 in reversed(range(lo + p, hi, b)):
            for j0 in reversed(range(lo, lo + p, b)):
                tg, layers = self.pair(lo, hi, p, b, i0, j0)
                for L in reversed(layers):
                    yield self._map(tg, L)
                del tg, layers
        yield from self.reverse(lo + p, hi)
        yield from self.reverse(lo, lo + p)


def transpose_layer(L):
    """Same rule as allength.compiler.transpose_word for one layer."""
    return [(d if s is not None else None, s if s is not None else d, c) for s, d, c in L]


# --------------------------------------------------------------------------- hashing / apply
def cval(c, p):
    return c.v if type(c) is GF else c % p


def layer_digest(L, p):
    raw = b"".join(RAW.pack(-1 if s is None else s, d, cval(c, p)) for s, d, c in L)
    return int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), "little") % MERSENNE


def sha_update(sha, idx, L, p):
    sha.update(b"".join(GATE.pack(idx, -1 if s is None else s, d, cval(c, p)) for s, d, c in L))


class Pass:
    def __init__(self, name):
        self.name = name
        self.sha = hashlib.sha256()
        self.layers = 0
        self.shears = 0
        self.scales = 0
        self.poly = 0
        self.emitted_order_poly = 0
        self.t = 0.0


def apply_stream(stream, ys, p, P, transpose, reverse, corrupt_at=None, total=None, log=None,
                 every=1_000_000):
    """Apply a layer stream to all vectors (ints mod p). `stream` yields FORWARD-word layers
    (in forward or reverse order); if `transpose`, ends are swapped before applying."""
    t0 = time.time()
    Bk = 1
    fidx = (total - 1) if (reverse and total is not None) else 0
    for L in stream:
        if corrupt_at is not None and fidx == corrupt_at:
            L = list(L)
            for k, (s, d, c) in enumerate(L):
                if s is not None:
                    L[k] = (s, d, c + 1)
                    break
        dg = layer_digest(L, p)
        # Unconditional Horner accumulation fingerprints the actual emitted order,
        # unlike P.poly, whose convention depends on the requested reverse mode.
        P.emitted_order_poly = (P.emitted_order_poly * POLY_B + dg) % MERSENNE
        if reverse:
            P.poly = (P.poly * POLY_B + dg) % MERSENNE
        else:
            P.poly = (P.poly + dg * Bk) % MERSENNE
            Bk = Bk * POLY_B % MERSENNE
        A = transpose_layer(L) if transpose else L
        sha_update(P.sha, P.layers, A, p)
        for s, d, c in A:
            cv = cval(c, p)
            if s is None:
                P.scales += 1
                for y in ys:
                    y[d] = y[d] * cv % p
            else:
                P.shears += 1
                for y in ys:
                    y[d] = (y[d] + cv * y[s]) % p
        P.layers += 1
        fidx += -1 if reverse else 1
        if log and P.layers % every == 0:
            log(f"  [{P.name}] layers={P.layers} gates={P.shears + P.scales} "
                f"t={time.time() - t0:.0f}s rss={cur_rss_mb():.0f}MB maxrss={rss_mb():.0f}MB")
    if reverse and total is not None:
        assert fidx == -1, f"reverse stream length mismatch (end index {fidx})"
    P.t = time.time() - t0
    return ys


# --------------------------------------------------------------------------- materialised
def materialised(F, h, R, xs_gf, Dh, Dmid, corrupt):
    """Project builder + transpose_word + exact_check_dft-style GF apply (validation only)."""
    from allength.compiler import transpose_word
    from allength.generic_word import g_block_word_local
    p = F.p
    word = g_block_word_local(F, h, R)
    if corrupt:                                  # identical to tests/test_exact_field.py
        for L in word[len(word) // 2:]:
            for k, (s, d, c) in enumerate(L):
                if s is not None:
                    L[k] = (s, d, c + 1)
                    break
            else:
                continue
            break
    tword = transpose_word(word)
    sw, st = hashlib.sha256(), hashlib.sha256()
    for i, L in enumerate(word):
        sha_update(sw, i, L, p)
    for i, L in enumerate(tword):
        sha_update(st, i, L, p)

    def apply_word(y, w):
        for L in w:
            for s, d, c in L:
                if s is None:
                    y[d] = c * y[d]
                else:
                    y[d] = y[d] + c * y[s]
        return y
    outs = []
    for x in xs_gf:
        y = [Dh[i] * x[i] for i in range(R)]
        y = apply_word(y, tword)
        y = [Dmid[i] * y[i] for i in range(R)]
        y = apply_word(y, word)
        y = [Dh[i] * y[i] for i in range(R)]
        outs.append([e.v for e in y])
    return {"layers": len(word), "sha_word": sw.hexdigest(), "sha_tword": st.hexdigest(),
            "outputs": outs}


def corrupt_target(ws, p):
    """Count forward layers; target = first layer with a shear at index >= total // 2."""
    layers = list(ws.forward())                  # small r only
    total = len(layers)
    for i in range(total // 2, total):
        if any(s is not None for s, _, _ in layers[i]):
            return total, i
    raise RuntimeError("no shear in second half")


# --------------------------------------------------------------------------- main
def check_prime(R, N, idx, nvec, basis, do_mat, corrupt, log):
    t0 = time.time()
    F = PrimeField(N, index=idx)
    p = F.p
    Dh, Dmid, h = fourier(F, R)
    ws = WordStream(F, h, R)
    rnd = random.Random(R)                       # as exact_check_dft.check (fresh per prime)
    if basis:
        xs = [[1 if t == j else 0 for t in range(R)] for j in range(R)]
        vec_kind = f"basis({R})"
    else:
        xs = [[rnd.randrange(p) for _ in range(R)] for _ in range(nvec)]
        vec_kind = f"random({nvec}), seed random.Random({R}).randrange(p)"
    vec_sha = hashlib.sha256(repr(xs).encode()).hexdigest()
    Dh_i, Dmid_i = [e.v for e in Dh], [e.v for e in Dmid]
    total = target = None
    if corrupt:
        total, target = corrupt_target(ws, p)
        log(f"  corruption: forward layer {target} of {total} (first shear coef +1)")
    log(f"r={R} prime#{idx} p={p} (p = 1 mod {N}) vectors={vec_kind} vec_sha={vec_sha[:16]}")
    ys = [[Dh_i[i] * x[i] % p for i in range(R)] for x in xs]
    PT = Pass("transpose")
    if BAD_TRANSPOSE == "noreverse":             # negative control: wrong layer order
        apply_stream(ws.forward(), ys, p, PT, transpose=True, reverse=False, log=log)
    elif BAD_TRANSPOSE == "noswap":              # negative control: ends not swapped
        apply_stream(ws.reverse(), ys, p, PT, transpose=False, reverse=True, log=log)
    else:
        apply_stream(ws.reverse(), ys, p, PT, transpose=True, reverse=True,
                     corrupt_at=target, total=total, log=log)
    log(f"  transpose pass: layers={PT.layers} gates={PT.shears + PT.scales} t={PT.t:.0f}s "
        f"maxrss={rss_mb():.0f}MB")
    ys = [[Dmid_i[i] * y[i] % p for i in range(R)] for y in ys]
    PF = Pass("forward")
    apply_stream(ws.forward(), ys, p, PF, transpose=False, reverse=False,
                 corrupt_at=target, total=total, log=log)
    log(f"  forward pass:   layers={PF.layers} gates={PF.shears + PF.scales} t={PF.t:.0f}s "
        f"maxrss={rss_mb():.0f}MB")
    ys = [[Dh_i[i] * y[i] % p for i in range(R)] for y in ys]
    upow = [pow(F.omega, (k * (N // R)) % N, p) for k in range(R)]
    checked = mismatches = 0
    for x, y in zip(xs, ys):
        for i in range(R):
            want = 0
            for j in range(R):
                want += upow[(i * j) % R] * x[j]
            checked += 1
            if y[i] != want % p:
                mismatches += 1
    res = {
        "r": R, "N": N, "prime_index": idx, "p": p, "omega": F.omega,
        "vectors": vec_kind, "n_vectors": len(xs), "vectors_sha256": vec_sha,
        "strict_KEEP_ZERO_MULTIPLIERS": CD.KEEP_ZERO_MULTIPLIERS,
        "corrupted": bool(corrupt), "bad_transpose": BAD_TRANSPOSE, "corrupt_forward_layer": target,
        "entries_checked": checked, "mismatches": mismatches,
        "layers_word": PF.layers, "layers_tword": PT.layers,
        "gates_word": {"shears": PF.shears, "scales": PF.scales},
        "gates_tword": {"shears": PT.shears, "scales": PT.scales},
        "gates_processed_total": PF.shears + PF.scales + PT.shears + PT.scales,
        "gate_applications_total": (PF.shears + PF.scales + PT.shears + PT.scales) * len(xs),
        "sha256_word_stream": PF.sha.hexdigest(), "sha256_tword_stream": PT.sha.hexdigest(),
        "polyhash_forward": hex(PF.poly), "polyhash_reverse": hex(PT.emitted_order_poly),
        "polyhash_transpose_input_emission_horner": hex(PT.emitted_order_poly),
        "transpose_input_layer_order": "forward" if BAD_TRANSPOSE == "noreverse" else "reverse",
        "transpose_swaps_shear_ends": BAD_TRANSPOSE != "noswap",
        "reverse_equals_reversed_forward": PF.poly == PT.emitted_order_poly and PF.layers == PT.layers,
        "pairs_built_both_passes": ws.pairs_built, "max_pair_dag_gates": ws.max_pair_gates,
        "max_pair_replay_layers": ws.max_pair_layers,
        "time_transpose_s": round(PT.t, 1), "time_forward_s": round(PF.t, 1),
        "time_total_s": round(time.time() - t0, 1), "maxrss_mb_so_far": round(rss_mb(), 1),
        "outputs_sha256": hashlib.sha256(repr(ys).encode()).hexdigest(),
    }
    if do_mat:
        xs_gf = [[F.from_int(v) for v in x] for x in xs]
        m = materialised(F, h, R, xs_gf, Dh, Dmid, corrupt)
        res["materialised"] = {
            "layers": m["layers"],
            "sha_word_equal": m["sha_word"] == res["sha256_word_stream"],
            "sha_tword_equal": m["sha_tword"] == res["sha256_tword_stream"],
            "outputs_equal": m["outputs"] == ys,
            "sha256_word": m["sha_word"], "sha256_tword": m["sha_tword"],
        }
        log(f"  materialised: layers={m['layers']} sha_word_equal={res['materialised']['sha_word_equal']} "
            f"sha_tword_equal={res['materialised']['sha_tword_equal']} "
            f"outputs_equal={res['materialised']['outputs_equal']}")
    log(f"exact-stream r={R} F_p p={p} {vec_kind.split(',')[0]}: entries checked={checked} "
        f"mismatches={mismatches} layers={PF.layers} gates/word={PF.shears + PF.scales} "
        f"rev==fwd:{res['reverse_equals_reversed_forward']} total={res['time_total_s']:.0f}s "
        f"maxrss={rss_mb():.0f}MB")
    return res


def main():
    a = _ARGS if _ARGS is not None else argument_parser().parse_args()
    global BAD_TRANSPOSE
    BAD_TRANSPOSE = a.bad_transpose
    R = a.R
    N = lcm(R, 2 ** math.ceil(math.log2(8 * R)))

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    log(f"start r={R} N={N} primes={a.primes} nvec={a.nvec} basis={a.basis} "
        f"mat={a.materialise_check} corrupt={a.corrupt} python={platform.python_version()} "
        f"pid={os.getpid()} strict={CD.KEEP_ZERO_MULTIPLIERS}")
    source_start = source_snapshot()
    t0 = time.time()
    runs = [check_prime(R, N, k, a.nvec, a.basis, a.materialise_check, a.corrupt, log)
            for k in range(a.first_prime, a.first_prime + a.primes)]
    source_end = source_snapshot()
    out = {"script": os.path.abspath(__file__), "project": PROJECT, "argv": sys.argv,
           "provenance": {"checker_version": CHECKER_VERSION, "schema_version": 2,
                          "method": "SHA-256 of on-disk sources at start/end; not bytecode attestation",
                          "start": source_start, "end": source_end,
                          "sources_unchanged_between_snapshots": sources_unchanged(source_start, source_end)},
           "python": sys.version, "r": R, "N": N, "runs": runs,
           "mismatches_total": sum(x["mismatches"] for x in runs),
           "wall_s": round(time.time() - t0, 1), "maxrss_mb": round(rss_mb(), 1)}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    log(f"done r={R} mismatches_total={out['mismatches_total']} wall={out['wall_s']}s "
        f"maxrss={out['maxrss_mb']}MB")
    sys.exit(0 if (a.corrupt or a.bad_transpose or out["mismatches_total"] == 0) else 1)


if __name__ == "__main__":
    main()
