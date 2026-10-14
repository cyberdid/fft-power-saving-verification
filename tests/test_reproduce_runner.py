"""Run isolation, required command coverage and real subprocess resume checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/'paper/scripts/reproduce.py'


class ReproduceRunner(unittest.TestCase):
    def test_invalid_mode_does_not_touch_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'run'; out.mkdir()
            sentinel = out/'keep'; sentinel.write_text('keep')
            proc = subprocess.run([sys.executable,str(RUNNER),'oops','--output-dir',str(out)],capture_output=True)
            self.assertEqual(proc.returncode,2)
            self.assertEqual(sentinel.read_text(),'keep')

    def test_full_plan_contains_previously_missing_published_jobs(self):
        proc = subprocess.run([sys.executable,str(RUNNER),'full','--list'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        plan=json.loads(proc.stdout); commands=[' '.join(job['argv']) for job in plan['jobs']]
        for fragment in ('exact_check_dft.py 32 cyclo','exact_check_dft.py 4 cyclo --cslots',
                         'exact_check_dft.py 16 cyclo --cslots','exact_check_dft.py 64 prime 1 0 --cslots',
                         'exact_check_dft.py 128 prime 1 2 --cslots','mp_check_cslots.py 256 550 1',
                         'mp_check_toeplitz_stream.py 1024 700 3','timing.py 512'):
            self.assertTrue(any(fragment in c for c in commands),fragment)
        self.assertFalse(plan['omitted'])
        self.assertTrue(any('unittest discover' in c for c in commands))
        self.assertTrue(any('collect_v6.py --log-dir' in c for c in commands))
        figure=next(c for c in commands if 'make_figures.py' in c)
        self.assertIn('--data-dir',figure)
        self.assertNotIn('paper/data',figure)

    def test_quick_plan_declares_omissions(self):
        proc=subprocess.run([sys.executable,str(RUNNER),'quick','--list'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(len(json.loads(proc.stdout)['omitted']),6)

    def test_real_jobs_preserve_successes_and_retry_failures(self):
        spec=importlib.util.spec_from_file_location('reproduce',RUNNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);source=out/'source';source.mkdir();(out/'raw').mkdir()
            counter=source/'counter'
            jobs=[{'name':'first','argv':[sys.executable,'-c',"from pathlib import Path; p=Path('counter'); p.write_text(p.read_text()+'x' if p.exists() else 'x')"]},
                  {'name':'second','argv':[sys.executable,'-c',"from pathlib import Path; raise SystemExit(0 if Path('allow').exists() else 3)"]}]
            manifest={'jobs':{},'status':'running'}
            with self.assertRaises(RuntimeError):module.execute_jobs(source,out,jobs,manifest,dict(os.environ))
            self.assertEqual(counter.read_text(),'x')
            (source/'allow').touch()
            module.execute_jobs(source,out,jobs,manifest,dict(os.environ))
            self.assertEqual(counter.read_text(),'x')
            self.assertEqual(manifest['jobs']['second']['returncode'],0)
            self.assertTrue(manifest['jobs']['first']['log_sha256'])

    def test_resume_rejects_modified_generated_data(self):
        spec=importlib.util.spec_from_file_location('reproduce_artifacts',RUNNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);source=out/'source';source.mkdir();(out/'raw').mkdir()
            data=out/'data'
            code="from pathlib import Path; d=Path("+repr(str(data))+"); d.mkdir(exist_ok=True); (d/'table.csv').write_text('original')"
            jobs=[{'name':'collect','argv':[sys.executable,'-c',code],'output_dirs':[str(data)]}]
            manifest={'jobs':{},'status':'running'}
            module.execute_jobs(source,out,jobs,manifest,dict(os.environ))
            (data/'table.csv').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'changed/missing output'):
                module.execute_jobs(source,out,jobs,manifest,dict(os.environ))

    def test_source_inventory_rejects_new_importable_file(self):
        spec=importlib.util.spec_from_file_location('reproduce_source',RUNNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            source=Path(td);(source/'original.py').write_text('x=1')
            files={'original.py':module.sha(source/'original.py')}
            module.verify_source(source,files,set())
            (source/'numpy.py').write_text('raise RuntimeError("shadowed")')
            with self.assertRaisesRegex(ValueError,'source inventory'):
                module.verify_source(source,files,set())

    def test_source_inventory_allows_missing_incomplete_generated_output(self):
        spec=importlib.util.spec_from_file_location('reproduce_retry',RUNNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            source=Path(td);(source/'original.py').write_text('x=1')
            files={'original.py':module.sha(source/'original.py')}
            module.verify_source(source,files,{'data/generated.pkl'})

    def test_snapshot_contains_root_modules_required_by_full_tests(self):
        spec=importlib.util.spec_from_file_location('reproduce_snapshot',RUNNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            source,_=module.snapshot(Path(td))
            proc=subprocess.run([sys.executable,'-c','import numerical, exact, precision_control; from allength import dft_all, assembly'],cwd=source,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)


if __name__=='__main__':unittest.main()
