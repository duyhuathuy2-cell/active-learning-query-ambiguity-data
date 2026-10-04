# Scientific reproduction

Run from the repository root. Install the cached-validation dependencies in `requirements/checks.txt`, then run `python reproduction/validate.py --full`. This validates archived predictions and results without new neural inference.

Scientific implementations are in `code/`; recorded data are in `../data/`. Full inference uses `requirements/inference.txt` and separately obtained official checkpoints described in `../models/README.md`. The original manuscript source is supplied separately to the journal. `compile_manuscript.py` requires that source under `KBS_submission/`.

The final plotting entry point is `generate_figures.py`; source mappings and statistical definitions are recorded in `../data/figure_sources.md`. Third-party notices are in `licenses/`.
