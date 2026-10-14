"""Negative controls for the publication data gate."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionVerifier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='verifier-test-')
        self.data = Path(self.tmp.name) / 'data'
        shutil.copytree(ROOT / 'paper/data', self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self):
        return subprocess.run([sys.executable, str(ROOT/'paper/scripts/verify_tables.py'),
                               '--data-dir', str(self.data)], cwd=ROOT,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                              capture_output=True, text=True)

    def rewrite(self, name, mutate):
        path = self.data/name
        with path.open() as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        rows = mutate(rows)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_reference_passes(self):
        proc = self.gate()
        self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)

    def test_missing_precision_results_fail(self):
        self.rewrite('v6/mp_dft.csv', lambda rows: [r for r in rows if r['r']!='128'])
        proc = self.gate()
        self.assertNotEqual(proc.returncode, 0, proc.stdout)

    def test_corrupt_float32_fails(self):
        def mutate(rows):
            for row in rows:
                if row['kind']=='f64_dft' and row['r']=='32':
                    row['max_rel_err']='1e100'
            return rows
        self.rewrite('dft_precision.csv', mutate)
        self.assertNotEqual(self.gate().returncode, 0)

    def test_corrupt_intermediate_depth_fails(self):
        def mutate(rows):
            for row in rows:
                if row['r']=='336' and row['rule']=='adaptive':
                    row['depth']='999999999'
            return rows
        self.rewrite('strict_tables.csv', mutate)
        self.assertNotEqual(self.gate().returncode, 0)

    def test_missing_cslots_fails(self):
        (self.data/'v6/cslots.csv').unlink()
        self.assertNotEqual(self.gate().returncode, 0)

    def test_nonfinite_precision_fails(self):
        self.rewrite('v6/mp_dft.csv', lambda rows: [dict(r, max_rel_err='nan') for r in rows])
        self.assertNotEqual(self.gate().returncode, 0)

    def test_duplicate_precision_key_fails(self):
        self.rewrite('v6/mp_dft.csv',lambda rows:rows+[dict(rows[0])])
        proc=self.gate()
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('duplicate key',proc.stdout)

    def test_nonfinite_condition_fails(self):
        self.rewrite('v6/cond.csv',lambda rows:[dict(row,log10_condT='nan') for row in rows])
        proc=self.gate()
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('condition T',proc.stdout)

    def test_random_vectors_do_not_count_as_full_matrix(self):
        self.rewrite('v6/exact.csv',lambda rows:[dict(row,vectors='random') if row['field'].startswith('F_p ') and int(row['r'])<=128 else row for row in rows])
        proc=self.gate()
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('Fp vector coverage',proc.stdout)
        self.assertIn('Fp C-slots',proc.stdout)

    def test_wrong_cyclotomic_field_metadata_fails(self):
        self.rewrite('v6/exact.csv',lambda rows:[dict(row,field='Q(zeta_128) [degree 64]') if row['r']=='32' and row['field'].startswith('Q(') else row for row in rows])
        proc=self.gate()
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('exact field order',proc.stdout)

    def test_three_labels_of_one_prime_do_not_count_as_three_primes(self):
        def mutate(rows):
            selected=[row for row in rows if row['r']=='32' and row['path']=='word' and row['field'].startswith('F_p ')]
            import re
            prime=re.search(r'p=(\d+)',selected[0]['field']).group(1)
            for index,row in enumerate(selected):
                row['field']=re.sub(r'p=\d+','p='+('0'*index)+prime,row['field'])
            return rows
        self.rewrite('v6/exact.csv',mutate)
        proc=self.gate()
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('Fp distinct primes r=32',proc.stdout)


if __name__ == '__main__':
    unittest.main()
