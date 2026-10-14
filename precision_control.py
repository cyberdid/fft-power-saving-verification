"""High-precision control of the local Newton identity, not a fast FFT."""
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np


def prepare(n):
    if n < 1:
        raise ValueError('positive length required')
    u = mp.exp(-2*mp.j*mp.pi/n)
    roots = [u**j for j in range(n)]
    H = [mp.mpc(1)]
    for j in range(1, n):
        H.append(H[-1]*(1-roots[j]))
    N = [[(-1)**k*u**(k*(k-1)//2)*H[i]/H[i-k]
          for k in range(i+1)] for i in range(n)]
    return roots, N


def transform(x, N):
    n = len(x)
    middle = [sum((N[i][k]*x[i] for i in range(k, n)), mp.mpc(0))/N[k][k]
              for k in range(n)]
    return [sum((N[i][k]*middle[k] for k in range(i+1)), mp.mpc(0))
            for i in range(n)]


def direct(x, roots):
    n = len(x)
    return [sum((roots[(i*j) % n]*x[j] for j in range(n)), mp.mpc(0))
            for i in range(n)]


def relative_error(actual, reference):
    return mp.sqrt(sum(abs(a-b)**2 for a, b in zip(actual, reference)) /
                   sum(abs(b)**2 for b in reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be absent or empty; refusing to overwrite')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, outputs = [], {}
    with np.load(args.input, allow_pickle=False) as archive:
        for digits in (50, 100):
            with mp.workdps(digits):
                for n in (32, 64, 128):
                    started = time.perf_counter()
                    roots, N = prepare(n)
                    setup = time.perf_counter()-started
                    for case in ('white_noise', 'weak_near_strong'):
                        # mpmath converts the already-recorded binary64 values;
                        # it does not generate a new, higher-precision signal.
                        values = archive[f'n{n}__{case}__windowed_iq']
                        x = [mp.mpc(float(v.real), float(v.imag)) for v in values]
                        started = time.perf_counter()
                        reference = direct(x, roots)
                        direct_seconds = time.perf_counter()-started
                        started = time.perf_counter()
                        actual = transform(x, N)
                        newton_seconds = time.perf_counter()-started
                        error = relative_error(actual, reference)
                        binary64 = archive[f'n{n}__{case}__paper_local_newton']
                        binary64_mp = [mp.mpc(float(v.real), float(v.imag)) for v in binary64]
                        row = {'decimal_digits': digits, 'n': n, 'case': case,
                               'relative_l2_error': mp.nstr(error, digits),
                               'binary64_relative_l2_vs_mp_direct': mp.nstr(relative_error(binary64_mp, reference), 20),
                               'status': 'PASS' if error <= mp.mpf('1e-10') else 'FAIL_NUMERICAL',
                               'preparation_seconds': setup, 'direct_seconds': direct_seconds,
                               'newton_seconds': newton_seconds}
                        rows.append(row)
                        key = f'dps{digits}__n{n}__{case}'
                        outputs[key] = {method: [[mp.nstr(v.real, digits), mp.nstr(v.imag, digits)]
                                                for v in data]
                                        for method, data in [('direct_dft', reference), ('newton', actual)]}
                        print(f'{key}: {row["status"]}, relative L2={mp.nstr(error, 6)}', flush=True)
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'high-precision local identity control; not the full new FFT',
                'input_path': str(args.input.resolve()),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'environment': {'python': platform.python_version(), 'mpmath': mp.__version__, 'numpy': np.__version__},
                'tolerance': '1e-10', 'rows': rows, 'outputs': outputs,
                'timing_note': 'Single control calls, not a repeated performance benchmark.'}
    (args.output/'precision.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    with (args.output/'precision.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ['# Контроль Newton factorization з підвищеною точністю', '',
              'Ті самі збережені windowed synthetic IQ; контрольний метод — прямий DFT у mpmath.',
              'Негативний знак DFT, звичайний transpose, не conjugate transpose.', '',
              '| Знаків | N | Сигнал | Relative L2 error | Статус |',
              '|---:|---:|---|---:|---|']
    report += [f'| {r["decimal_digits"]} | {r["n"]} | {r["case"]} | {float(r["relative_l2_error"]):.3e} | {r["status"]} |'
               for r in rows]
    report += ['', 'PASS: relative L2 ≤ 1e-10. Precision — робочі десяткові знаки, не exact arithmetic.',
               'Якщо помилка зменшується з підвищенням точності, це свідчення втрати точності у complex128,',
               'а не практичного прискорення або спростування exact-complex theorem.',
               'Прямий контроль використовує ту саму обчислену root of unity, тому окремі literal DFT fixtures',
               'основного набору перевіряють конвенцію знака незалежно.',
               'Тут O(N²) Python-прототипи; single-call timings — службові, не конкурентний FFT benchmark.', '']
    (args.output/'REPORT.uk.md').write_text('\n'.join(report))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.output.iterdir() if p.is_file()}
    (args.output/'artifact_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    if any(r['status'] != 'PASS' for r in rows):
        raise SystemExit('Some high-precision controls failed; see saved report.')


if __name__ == '__main__':
    main()
