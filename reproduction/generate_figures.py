"""Regenerate only the eight figures cited by the manuscript."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
(ROOT/'output').mkdir(exist_ok=True)
for name in ['publication_figures']:
    subprocess.run([sys.executable, str(ROOT/'code'/(name+'.py'))], check=True)
print('Eight manuscript figures generated as PDF, SVG and 600-dpi PNG.')
