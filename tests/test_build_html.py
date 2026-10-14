"""Keep isolated publication HTML tied to the selected source and figures."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class IsolatedHTML(unittest.TestCase):
    def test_print_uses_selected_source_figures_and_lettered_caption(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);src=path/'article.txt';out=path/'print.html';fig=path/'new figures'
            src.write_text('TEST TITLE\n\nРОЗДІЛ\n\nТаблиця 2d. TEST CAPTION\n| a | b |\n|---|---|\n| 1 | 2 |\n\nРисунок 1 (fig1_depth.pdf). TEST FIGURE\n')
            proc=subprocess.run([sys.executable,str(ROOT/'paper/scripts/build_html.py'),'print',str(out),'--source',str(src),'--fig-dir',str(fig)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            html=out.read_text()
            self.assertIn('TEST TITLE',html)
            self.assertIn((fig/'fig1_depth.png').as_uri(),html)
            self.assertIn('class="caption">Таблиця 2d.',html)


if __name__=='__main__':unittest.main()
