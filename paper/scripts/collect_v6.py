"""Collect one run's raw logs into publication CSVs, without reading another run.

Default invocation explicitly collects the archived v6/strict_parts directories and
legacy precision CSV. For a fresh run use --log-dir RUN/raw --output-dir RUN/data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
NUMBER = r'([-+\d.eE]+)'
# Runner hashes administrative logs separately after they finish. The collector must
# not record its own partially written stdout or treat test output as experimental data.
ADMIN_LOGS = {'collect.log','verify.log','figures.log','tests.log'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def diagonal_extrema(r):
    import mpmath as mp
    with mp.workdps(60):
        values=[mp.mpf(0)]
        for k in range(1,r):values.append(values[-1]+mp.log10(2*mp.sin(mp.pi*k/r)))
        return f'{float(max(values)):.1f}',f'{float(-min(values)):.1f}'


def collect(log_dirs, output_dir, legacy_precision=None):
    output_dir = Path(output_dir)
    tables = {}
    def add(name, fields, values):
        table = tables.setdefault(name, {'fields': fields, 'rows': []})
        table['rows'].append(tuple(str(x) for x in values))
    num = NUMBER
    mp_rx = re.compile(rf'^mp  r=(\d+) dps=(\d+) (\w+)\((\d+)\): max rel err={num} median={num} digits lost={num} log10 max\|H\|={num} log10 max\|1/H\|={num}')
    cs_rx = re.compile(rf'^cslots r=(\d+) dps=(\d+) (\w+)\((\d+)\): shear blocks per vector=(\d+) .*?max rel err={num} digits lost={num}(.*)$')
    ex_rx = re.compile(r'^exact r=(\d+) (.+?) (word|C-slots\+conj-DAG) (\w+)\((\d+)\): entries checked=(\d+) mismatches=(\d+)')
    toe_rx = re.compile(rf'^toeplitz strict r=(\d+) dps=(\d+) vectors=(\d+): max rel err={num} min={num} digits lost={num}')
    fl_rx = re.compile(rf'^f64 r=(\d+) (\w+)\((\d+)\): max rel err={num} median={num}')
    cond_rx = re.compile(rf'^r=(\d+): log10 cond2\(T\)={num} log10 smax={num} log10 smin={num} log10 cond2\(Dh\)={num}')
    files = sorted({p for directory in map(Path, log_dirs) for p in directory.rglob('*') if p.is_file() and p.name not in ADMIN_LOGS and p.suffix in ('.log','.txt')})
    precision = []
    float_details = {}
    growth = {}
    for path in files:
        stage_r = None
        peak_r, peak_stage, total, by_phase, by_level = None, None, None, {}, {}
        def flush_peak():
            if peak_stage is not None and total is not None:
                add('v6/causal_levels.csv', ['r','stage','growth','chunk','direct','leaf','levels'],
                    [peak_r,peak_stage,total,by_phase.get('chunk',0),by_phase.get('direct',0),by_phase.get('leaf',0),json.dumps(by_level,sort_keys=True)])
        for line in path.read_text().splitlines():
            if match := mp_rx.match(line):
                g = match.groups()
                add('v6/mp_dft.csv', ['r','dps','vectors','n','max_rel_err','median','digits_lost','log10_maxH','log10_maxinvH'], g)
                precision.append(('mp_dft',g[0],g[1],g[2] if g[2]=='basis' else g[2]+g[3],g[4],g[6],g[7]))
            if match := cs_rx.match(line):
                g = match.groups()
                add('v6/cslots.csv',['r','dps','vectors','n','blocks_per_vector','max_rel_err','digits_lost','conj'],g[:7]+('conj-DAG' if 'conj-DAG' in g[7] else 'abs',))
            if match := ex_rx.match(line):
                add('v6/exact.csv',['r','field','path','vectors','n','entries','mismatches'],match.groups())
            if match := toe_rx.match(line):
                g=match.groups()
                growth_match=re.search(r'log10 max\|g\|=([-\d.]+)',line)
                add('v6/toeplitz.csv',['r','dps','vectors','max_rel_err','min_rel_err','digits_lost','log10_maxg'],g+(growth_match[1] if growth_match else '',))
                precision.append(('mp_toeplitz',g[0],g[1],'random'+g[2],g[3],g[5],''))
            if match := fl_rx.match(line):
                g=match.groups()
                float_details[g[0]] = (g[1],g[2],g[4])
                precision.append(('f64_dft',g[0],'16',g[1] if g[1]=='basis' else g[1]+g[2],g[3],'',''))
            if match := cond_rx.match(line):
                add('v6/cond.csv',['r','log10_condT','log10_smax','log10_smin','log10_condDh'],match.groups())
            if match := re.match(r'^r=(\d+) dps=(\d+) \(reference',line):
                stage_r=match.group(1)
            if stage_r and (match := re.match(rf'^(\d) (\S+)\s+{num}\s+{num}\s+{num}\s+{num}',line)):
                add('v6/stagewise.csv',['r','stage','name','local_lost','cumul_lost','log10_growth','log10_norm'],(stage_r,)+match.groups())
            if match := re.match(r'^width-model r=(\d+) b=(\d+) depth=(\d+)',line):
                add('depth_model_widths.csv',['r','b','depth'],match.groups())
            if match := re.match(r'^2\^(\d+)\s+(\d+)\s+(\d+)\s+([-\d.]+)',line):
                e,b,depth,ratio=match.groups()
                add('depth_model_strict.csv',['e','r','b','depth','ratio'],[e,2**int(e),b,depth,ratio])
            if match := re.match(rf'^r=(\d+) gates=(\d+) build={num}s apply={num}s peak_rss={num}GB numpy_fft={num}us numpy_dense_toeplitz={num}us',line):
                add('v6/timing.csv',['r','gates','build_seconds','apply_seconds','peak_rss_gib','fft_us','dense_us'],match.groups())
            if match := re.match(rf'^r=(\d+): min\|root\|={num} .*?roots inside unit disk: (\d+) of (\d+); rel. residual at smallest roots <= {num}; predicted log10 max\|g\| ~ {num}, observed {num}; fitted slope {num} vs log10\(1/rho\) {num}',line):
                g=match.groups();growth[g[0]]=g[6]
                add('v6/roots.csv',['r','rho','inside','degree','residual','predicted_growth','growth','slope','reciprocal_slope'],g)
            if match := re.match(r'^r=(\d+) dps=\d+  \(charges',line):
                flush_peak();peak_r=match.group(1);peak_stage=None;total=None;by_phase={};by_level={}
            if match := re.match(r'^stage (T\^T|T): log10\(peak/input_inf\) = ([-\d.]+)',line):
                flush_peak();peak_stage,total=match.groups();by_phase={};by_level={}
            if peak_stage and (match := re.match(r'^\s+v=(\d+)\s+growth=\s*([-\d.]+)',line)):
                by_level[match.group(1)]=float(match.group(2))
            if peak_stage and (match := re.match(r'^\s+(chunk|direct|leaf)\s+growth=\s*([-\d.]+)',line)):
                by_phase[match.group(1)]=match.group(2)
            if match := re.match(r'^r=(\d+): max \|by rounds - sequential\| = (\S+)',line):
                add('v6/rounds.csv',['r','kind','value','layers','rounds'],[match[1],'difference',match[2],'',''])
            if match := re.match(r'^r=(\d+): layers=(\d+) rounds=(\d+) conflicts=(\d+)',line):
                add('v6/rounds.csv',['r','kind','value','layers','rounds'],[match[1],'conflicts',match[4],match[2],match[3]])
        flush_peak()
    for directory in map(Path,log_dirs):
        for path in directory.rglob('*.csv'):
            rr=read_csv(path)
            if not rr:continue
            if {'r','rule','input','b_adm','depth','shears','scales','ops','signature'} <= set(rr[0]):
                fields=['r','rule','input','b_adm','depth','layers','shears','scales','ops','signature']
                for row in rr:add('strict_tables.csv',fields,[row[k] for k in fields])
            elif {'r','rule','input','hash_types_rounds'} <= set(rr[0]):
                fields=['r','rule','input','depth','layers','shears','scales','hash']
                for row in rr:add('v6/topology.csv',fields,[row.get(k,row.get('hash_types_rounds')) for k in fields])
        for path in directory.rglob('phase-*.json'):
            obj=json.loads(path.read_text())
            if not isinstance(obj,dict) or 'baseline_lost' not in obj:continue
            hp=obj['high_precision_lost']
            add('v6/phase_precision.csv',['r','stage','dps','reference_dps','baseline_lost','chunk_high_lost','direct_high_lost','chunk_only_low_lost'],
                [obj['r'],obj['stage'],obj['dps'],obj['reference_dps'],obj['baseline_lost'],hp.get('chunk'),hp.get('direct'),obj['chunk_only_low_lost']])
    if legacy_precision:
        for row in read_csv(Path(legacy_precision)):
            if row['kind']=='mp_dft' and int(row['r']) in (8,16,32):
                r=row['r'];add('v6/mp_dft.csv',['r','dps','vectors','n','max_rel_err','median','digits_lost','log10_maxH','log10_maxinvH'],
                    [r,row['dps'],'basis',r,row['max_rel_err'],'',row['digits_lost'],row['log10_growth'],diagonal_extrema(int(r))[1]])
            if row['kind']=='f64_dft':
                precision.append(tuple(row[k] for k in ('kind','r','dps','vectors','max_rel_err','digits_lost','log10_growth')))
            if row['kind']=='mp_toeplitz':growth[row['r']]=row['log10_growth']
    for row in precision:
        if row[0]=='mp_toeplitz':row=row[:-1]+(growth.get(row[1],''),)
        add('dft_precision.csv',['kind','r','dps','vectors','max_rel_err','digits_lost','log10_growth','median'],row+(float_details.get(row[1],('','',''))[2] if row[0]=='f64_dft' else '',))
    outputs={}
    for name,table in sorted(tables.items()):
        path=output_dir/name;path.parent.mkdir(parents=True,exist_ok=True)
        unique=sorted(set(table['rows']))
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(table['fields']);w.writerows(unique)
        outputs[name]=len(unique)
    inputs = sorted({p for directory in map(Path,log_dirs) for p in directory.rglob('*') if p.is_file() and p.name not in ADMIN_LOGS and p.suffix in ('.log','.txt','.csv','.json')})
    return {'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'log_files':[str(p) for p in files],'output_rows':outputs,
            'legacy_precision':str(legacy_precision) if legacy_precision else None}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir',type=Path,action='append')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--legacy-precision',type=Path)
    args=parser.parse_args(argv)
    dirs=args.log_dir or [ROOT/'paper/data/v6',ROOT/'paper/data/strict_parts']
    out=args.output_dir or ROOT/'paper/data'
    legacy=args.legacy_precision
    if not args.log_dir and legacy is None:legacy=ROOT/'paper/data/dft_precision.csv'
    result=collect(dirs,out,legacy)
    out.mkdir(parents=True,exist_ok=True)
    (out/'collection.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,count in result['output_rows'].items():print(name,count)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
