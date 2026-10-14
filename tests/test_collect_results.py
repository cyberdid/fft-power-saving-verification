"""Verify collectors use the requested run and never overwrite archived sources."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CollectResults(unittest.TestCase):
    def test_requested_run_is_collected_into_its_own_directory(self):
        with tempfile.TemporaryDirectory(prefix='collect-test-') as td:
            root = Path(td)
            scripts = root/'project/paper/scripts'
            archive = root/'project/paper/data/v6'
            raw, dest = root/'new-run/raw', root/'new-run/data'
            for d in (scripts, archive, raw):
                d.mkdir(parents=True)
            shutil.copy2(ROOT/'paper/scripts/collect_v6.py', scripts/'collect_v6.py')
            sentinel = archive/'cslots.csv'
            sentinel.write_text('unchanged archive\n')
            (raw/'fresh.log').write_text(
                'cslots r=256 dps=550 random(1): shear blocks per vector=20205056 '
                '(each 4 monomial + 3 C slots) max rel err=2.38e-381 digits lost=169.4 conj-DAG\n'
                'f64 r=32 basis(32): max rel err=0.000336 median=3.43e-05\n')
            proc = subprocess.run([sys.executable, str(scripts/'collect_v6.py'),
                                   '--log-dir', str(raw), '--output-dir', str(dest)],
                                  capture_output=True, text=True,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
            self.assertEqual(sentinel.read_text(), 'unchanged archive\n')
            self.assertTrue((dest/'v6/cslots.csv').is_file())
            self.assertIn('256,550', (dest/'v6/cslots.csv').read_text())
            self.assertIn('f64_dft,32', (dest/'dft_precision.csv').read_text())

    def test_provenance_excludes_administrative_self_log(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/'raw';dest=Path(td)/'data';raw.mkdir()
            (raw/'experiment.log').write_text('f64 r=32 basis(32): max rel err=0.000336 median=3.43e-05\n')
            with (raw/'collect.log').open('w') as stream:
                proc=subprocess.run([sys.executable,str(ROOT/'paper/scripts/collect_v6.py'),'--log-dir',str(raw),'--output-dir',str(dest)],stdout=stream,stderr=subprocess.STDOUT)
            self.assertEqual(proc.returncode,0)
            import json,hashlib
            provenance=json.loads((dest/'collection.json').read_text())['input_sha256']
            self.assertNotIn(str(raw/'collect.log'),provenance)
            for file,digest in provenance.items():self.assertEqual(hashlib.sha256(Path(file).read_bytes()).hexdigest(),digest)


if __name__ == '__main__':
    unittest.main()
