# Evidence

Frozen records of the completed reproduction run and of the checks performed on
it. These files are kept **byte-for-byte as produced**; they are not edited,
because their SHA-256 values are recorded in `manifest.json` and in the
integrity checks.

## full-20261013-r3/

The completed `full` core run (started 2026-10-13, status `complete`,
135/135 steps, table validation PASS with 1330 checks).

| File / folder | Contents |
|---|---|
| `manifest.json` | Plan, commands, exit codes, timings, SHA-256 of the captured sources, of every job log (`log_sha256`), of job outputs and of `data/` and `figures/` (`artifact_sha256`) |
| `raw/` | Raw log of every job and per-job JSON outputs (163 files) |
| `data/` | CSVs aggregated from `raw/` by `collect_v6.py`, plus `collection.json` |
| `figures/` | Figures generated from `data/` by `make_figures.py` |
| `environment-freeze.txt` | `pip freeze` of the clean environment used by the run |
| `verification.json` | Result of `verify_tables.py --mode full` at the end of the run |

Notes:

- **Absolute paths.** `manifest.json`, `data/collection.json` and some logs
  contain absolute paths of the original machine on which the run was executed
  (interpreter, run directory, captured source directory). They are historical
  records, not paths you need; a new run on another machine records its own.
- **Hashes.** The SHA-256 values in `manifest.json` refer to the files in this
  directory: every `log_sha256` matches a file in `raw/`, and every
  `artifact_sha256` entry matches the corresponding file under `data/` and
  `figures/`. Job outputs (`output_sha256`) are listed with original absolute
  paths; strip the run-directory prefix to find them here — the two gate-count
  tables written into the captured source correspond to `data/*.pkl` at the
  repository root.
- **Captured sources are omitted.** The run executed from a captured copy of the
  sources (`source/` in the run directory, hashed in `source_sha256`). That copy
  is not included here to avoid a second copy of the code; most files are
  identical to the current code in this repository, and the differences are the
  later edits of the manuscript, the aggregated CSVs, the validator, the figure,
  HTML/PDF build and runner scripts and one test. The numerical generators were
  not changed after the run.
- This copy cannot be resumed as a run on another machine. To reproduce, start a
  new run with `bash release/run_all.sh full --output-dir /new/path`.

## independent-checks/

JSON records of checks run on the completed run after it finished: integrity of
the run directory (`full-independent-integrity.json`), conjugate-coefficient
check of the C-slot logs (`full-conjugate-check.json`), table validation of the
current manuscript against the run data (`full-current-manuscript-verification.json`,
`v6.3-current-tables.json`), the final summary for v6.3 (`v6.3-final-checks.json`)
and a portable streamed exact checker at r = 16 (`v6.3-portable-checker-r16.json`).
They are records of commands actually run, not mathematical proofs, and may
contain absolute paths of the original machine.

`README.uk.md` is the Ukrainian version of these notes.

## Initial Git package check

`independent-checks/git-initial-validation.json` records the current export's
123 tests and 1531 full table checks after editorial updates to the validation
reference. The earlier v6.3 receipts retain their historical article/PDF hashes;
they are not rewritten to describe the later English draft or editorial text.
The completed core-run files are unchanged. The source snapshot of that run
remains omitted as described above.
