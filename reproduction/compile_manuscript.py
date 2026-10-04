"""Compile the current editable manuscript with an existing TeX installation."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'KBS_submission'
xelatex, bibtex = shutil.which('xelatex'), shutil.which('bibtex')
if not xelatex or not bibtex:
    raise SystemExit('XeLaTeX and BibTeX must be available in the current environment.')
commands = [[xelatex, '--disable-installer', '-interaction=nonstopmode', '-halt-on-error', 'manuscript.tex'],
            [bibtex, 'manuscript'],
            *[[xelatex, '--disable-installer', '-interaction=nonstopmode', '-halt-on-error', 'manuscript.tex']]*2]
for command in commands:
    result = subprocess.run(command, cwd=PAPER, capture_output=True)
    if result.returncode:
        print((result.stdout+result.stderr).decode('utf8', errors='replace'))
        raise SystemExit(result.returncode)
print('Manuscript compiled.')
