"""Validate saved scientific results without rerunning pretrained models."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Also check all saved confirmation inference matrices.')
args = parser.parse_args()
(ROOT / 'output').mkdir(exist_ok=True)
commands = [
    ['validate_benchmarks.py'],
    ['verify_identifiability.py'],
    ['validate_policy_controls.py'],
    ['validate_confirmation.py', *([] if args.full else ['--review'])],
    ['sensitivity_experiments.py', '--check-cached'],
]
for command in commands:
    print('Checking ' + command[0], flush=True)
    result = subprocess.run([sys.executable, str(ROOT/'code'/command[0]), *command[1:]], capture_output=True)
    if result.returncode:
        print((result.stdout + result.stderr).decode('utf8', errors='replace'))
        raise SystemExit(result.returncode)
print('All five scientific checks passed.', flush=True)
