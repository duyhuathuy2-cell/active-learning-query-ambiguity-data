"""Portable project-local configuration; install dependencies in your environment."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
TABPFN_CACHE = PROJECT / "models" / "tabpfn"
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(TABPFN_CACHE))
os.environ.setdefault("HF_HOME", str(PROJECT / "models" / "huggingface"))
