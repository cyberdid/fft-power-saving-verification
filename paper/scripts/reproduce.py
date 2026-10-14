"""Reproduce published experiments in an isolated, resumable run directory.

Every experiment runs in a source snapshot. Only this run's raw outputs feed its
CSV collector, validator and figures. Full mode covers the 135-step core grid;
the streamed F_p r=1024 check is a separately documented supplemental experiment.
Historical v4 comparison curves remain explicitly labelled archived references.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
WIDTHS = (64,128,256,288,320,336,352,384,512,640,768,1024)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_manifest(out, manifest):
    temporary=out/'manifest.pending.json'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(out/'manifest.json')


def plan(mode, out, python):
    out=Path(out);raw=out/'raw';jobs=[]
    def add(name,script,*args,outputs=(),output_dirs=()):
        jobs.append({'name':name,'argv':[str(python),'paper/scripts/'+script,*map(str,args)],
                     'outputs':list(map(str,outputs)), 'output_dirs':list(map(str,output_dirs))})
    jobs.append({'name':'tests','argv':[str(python),'-m','unittest','discover','-s','tests'],'outputs':[]})
    for strict in (False,True):
        suffix='_strict' if strict else ''
        if mode=='full' or not (ROOT/f'data/gate_count_table_b_le_1024{suffix}.pkl').exists():
            add('gate-table'+suffix,'build_gate_table.py',*(['--strict'] if strict else []),outputs=[out/'source'/f'data/gate_count_table_b_le_1024{suffix}.pkl'])
    for r in (4,7,8,9,12,15,16,32):add(f'exact-cyclo-{r}','exact_check_dft.py',r,'cyclo')
    for r in (4,8,16):add(f'exact-cyclo-cslots-{r}','exact_check_dft.py',r,'cyclo','--cslots')
    for r in (9,15,16,31,32,33,63,64,127,128):add(f'exact-prime-{r}','exact_check_dft.py',r,'prime',3)
    for r,n in ((255,3),(256,3),(512,2)):add(f'exact-prime-{r}','exact_check_dft.py',r,'prime',2,n)
    for r in (16,31,32,33,64,128):add(f'exact-prime-cslots-{r}','exact_check_dft.py',r,'prime',1,2 if r==128 else 0,'--cslots')
    for r,precisions,basis in ((7,(60,160),False),(8,(60,160),False),(9,(60,160),False),
            (15,(60,160),False),(16,(60,160),False),(31,(60,160),False),(32,(60,160),False),
            (33,(60,160),True),(63,(100,200),True),(64,(120,220),True),(96,(200,300),False),
            (127,(250,350),False),(128,(250,350),True),(256,(450,550),False)):
        for dps in precisions:add(f'dft-{r}-{dps}','mp_check_dft.py',r,dps,3,*(['--basis'] if basis else []))
    for r,dps in ((7,60),(9,60),(15,60),(31,60),(33,60),(63,100),(64,120),(64,220)):
        add(f'cslots-{r}-{dps}','mp_check_cslots.py',r,dps,3,'--basis','--conj-dag')
    for r,dps,n in ((128,250,3),(128,350,3),*(((256,450,1),(256,550,1)) if mode=='full' else ())):
        add(f'cslots-{r}-{dps}','mp_check_cslots.py',r,dps,n,'--conj-dag')
    for r in (8,16,24,32,48,64,96):add(f'float64-{r}','mp_check_dft.py',r,0,6,'--float')
    for r,dps in ((32,60),(64,120),(128,250)):
        add(f'stage-{r}','stagewise.py',r,dps,*(['--cond'] if r<=64 else []))
    for r,dps in ((32,40),(64,60),(128,100)):add(f'causal-{r}','causal_levels.py',r,dps)
    for r,dps in ((64,50),(128,90)):
        for stage in ('tword','word'):
            result=raw/f'phase-{r}-{stage}.json'
            add(f'phase-{r}-{stage}','phase_precision.py',r,dps,'--stage',stage,'--output-json',result,outputs=[result])
    for r,dps in ((16,60),(32,80),(48,100),(96,160),(128,200)):add(f'cond-{r}','cond_T.py',r,dps)
    for dps in (300,400):add(f'toeplitz-352-{dps}','mp_check_toeplitz.py',352,dps,3)
    if mode=='full':
        for r,precisions in ((640,(400,500)),(1024,(600,700))):
            for dps in precisions:add(f'toeplitz-{r}-{dps}','mp_check_toeplitz_stream.py',r,dps,3)
    add('roots','roots_h.py')
    for r in WIDTHS:
        result=raw/f'strict-{r}.csv';add(f'strict-{r}','strict_tables.py',result,r,outputs=[result])
        result=raw/f'topology-{r}.csv';add(f'topology-{r}','topology_audit.py',result,r,outputs=[result])
    add('depth-model','depth_exact_rule.py','--strict')
    add('rounds','rounds_check.py')
    for r in (64,128,256,512):add(f'timing-{r}','timing.py',r)
    add('collect','collect_v6.py','--log-dir',raw,'--output-dir',out/'data',output_dirs=[out/'data'])
    add('verify','verify_tables.py','--data-dir',out/'data','--mode',mode,'--output-json',out/'verification.json',outputs=[out/'verification.json'])
    add('figures','make_figures.py','--data-dir',out/'data','--fig-dir',out/'figures',
        outputs=[out/'figures'/f'{name}.{extension}' for name in ('fig1_depth','fig2_cost','fig3_log4_law','fig4_tradeoff_r512','fig5_precision','fig6_stagewise') for extension in ('png','pdf')],output_dirs=[out/'figures'])
    omitted=[] if mode=='full' else ['C-slots 256 / '+str(dps) for dps in (450,550)]+['Toeplitz '+str(r)+' / '+str(dps) for r,precisions in ((640,(400,500)),(1024,(600,700))) for dps in precisions]
    return {'mode':mode,'jobs':jobs,'omitted':omitted,
            'archived_references':['v4 zero-pruned comparison curves in figures (labelled historical)']}


def snapshot(out, generated=()):
    source=out/'source';source.mkdir()
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    for directory in ('allength','tests','data','paper/scripts','paper/data'):
        shutil.copytree(ROOT/directory,source/directory,ignore=ignore)
    for module in ROOT.glob('*.py'):shutil.copy2(module,source/module.name)
    shutil.copy2(ROOT/'paper/article-ua.txt',source/'paper/article-ua.txt')
    generated=set(generated)
    files={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file() and str(p.relative_to(source)) not in generated}
    return source,files


def verify_source(source,files,generated):
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if not set(files)<=actual<=set(files)|set(generated):raise ValueError('source inventory differs from captured source')
    for file,digest in files.items():
        if sha(source/file)!=digest:raise ValueError(f'source snapshot changed: {file}')


def execute_jobs(source,out,jobs,manifest,env):
    for job in jobs:
        name=job['name'];previous=manifest['jobs'].get(name)
        if previous and previous.get('returncode')==0:
            if previous['argv']!=job['argv']:raise RuntimeError(f'changed command: {name}')
            log=out/'raw'/f'{name}.log'
            if not log.exists() or sha(log)!=previous['log_sha256']:raise RuntimeError(f'changed/missing log: {name}')
            for file,digest in previous.get('output_sha256',{}).items():
                if not Path(file).exists() or sha(file)!=digest:raise RuntimeError(f'changed/missing output: {file}')
            for directory in job.get('output_dirs',[]):
                actual={str(p) for p in Path(directory).rglob('*') if p.is_file()}
                expected={p for p in previous.get('output_sha256',{}) if Path(p).is_relative_to(directory)}
                if actual!=expected:raise RuntimeError(f'changed/missing output directory: {directory}')
            continue
        for directory in job.get('output_dirs',[]):
            target=Path(directory).resolve()
            if target.parent!=out.resolve() or target.name not in ('data','figures'):
                raise RuntimeError(f'invalid generated directory: {target}')
            if target.exists():shutil.rmtree(target)
        for output in job.get('outputs',[]):Path(output).unlink(missing_ok=True)
        log=out/'raw'/f'{name}.log'
        entry={'argv':job['argv'],'started':timestamp(),'returncode':None}
        manifest['jobs'][name]=entry;manifest['current_job']=name;save_manifest(out,manifest)
        print(f'[{timestamp()}] {name}',flush=True)
        start=time.monotonic()
        with log.open('w') as stream:
            proc=subprocess.run(job['argv'],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
        produced=set(job.get('outputs',[]))
        for directory in job.get('output_dirs',[]):
            produced.update(str(p) for p in Path(directory).rglob('*') if p.is_file())
        entry.update(returncode=proc.returncode,finished=timestamp(),seconds=time.monotonic()-start,log_sha256=sha(log),
                     output_sha256={p:sha(p) for p in sorted(produced) if Path(p).is_file()})
        missing=[p for p in job.get('outputs',[]) if not Path(p).is_file()]+[p for p in job.get('output_dirs',[]) if not Path(p).is_dir()]
        if proc.returncode or missing:
            if missing and proc.returncode==0:entry['returncode']=1
            manifest['status']='failed';save_manifest(out,manifest)
            raise RuntimeError(f'{name} failed ({proc.returncode}); missing outputs={missing}; see {log}')
        save_manifest(out,manifest)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('quick','full'),nargs='?',default='quick')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--list',action='store_true',help='print command plan without creating files')
    args=parser.parse_args(argv)
    out=(args.output_dir or ROOT/'release/out'/('run-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])).absolute()
    python=str(Path(sys.executable).absolute())
    if args.list:
        print(json.dumps(plan(args.mode,out,python),indent=2));return 0
    if args.resume:
        if not (out/'manifest.json').is_file():parser.error('--resume requires an existing manifest')
        manifest=json.loads((out/'manifest.json').read_text());source=out/'source'
        if manifest['mode']!=args.mode:parser.error('mode differs from original run')
        try:verify_source(source,manifest['source_sha256'],manifest.get('generated_source_inputs',[]))
        except ValueError as error:parser.error(str(error))
        jobs=manifest['plan']['jobs']
    else:
        if out.exists():parser.error('output directory already exists; choose a new one or --resume')
        out.mkdir(parents=True);(out/'raw').mkdir()
        planned=plan(args.mode,out,python)
        generated={str(Path(p).relative_to(out/'source')) for job in planned['jobs'] for p in job.get('outputs',[]) if Path(p).is_relative_to(out/'source')}
        source,files=snapshot(out,generated)
        manifest={'mode':args.mode,'status':'running','started':timestamp(),'python':python,
                  'source_sha256':files,'plan':planned,'jobs':{},
                  'generated_source_inputs':sorted(generated)}
        jobs=planned['jobs'];save_manifest(out,manifest)
    (out/'tmp').mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             TMPDIR=str(out/'tmp'),MPLCONFIGDIR=str(out/'tmp/matplotlib'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if args.resume and python!=manifest['python']:parser.error('resume requires the original Python interpreter')
    freeze=subprocess.run([python,'-m','pip','freeze'],env=env,capture_output=True,text=True,check=True).stdout
    freeze_file=out/'environment-freeze.txt'
    if args.resume and freeze_file.read_text()!=freeze:parser.error('installed environment differs from original run')
    freeze_file.write_text(freeze)
    subprocess.run([python,'-m','pip','check'],env=env,check=True)
    manifest['status']='running';save_manifest(out,manifest)
    try:execute_jobs(source,out,jobs,manifest,env)
    except RuntimeError as error:
        print(error,file=sys.stderr);return 1
    verify_source(source,manifest['source_sha256'],manifest.get('generated_source_inputs',[]))
    # Coverage is supplied by the selected plan and gate; immutable source and raw hashes
    # make a later inspection independent of transient console output.
    manifest.update(status='complete',finished=timestamp(),current_job=None,
        artifact_sha256={str(p.relative_to(out)):sha(p) for folder in ('data','figures') for p in sorted((out/folder).rglob('*')) if p.is_file()})
    save_manifest(out,manifest)
    print(f'complete: {out}',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
