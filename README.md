# Active-learning query ambiguity: experimental data

Experimental data and scientific reproduction code for **Decision ambiguity in active learning: Set-valued coherence repair for query selection**.

- Creator: Zongzhi Guo, Northeastern University.
- Dataset title: Experimental data for Decision ambiguity in active learning: Set-valued coherence repair for query selection.
- Version: 1.0.0 (4 October 2026).
- Version record: [https://github.com/duyhuathuy2-cell/active-learning-query-ambiguity-data/releases/tag/v1.0.0](https://github.com/duyhuathuy2-cell/active-learning-query-ambiguity-data/releases/tag/v1.0.0).
- Data inventory: 4,959 files, 258.4 MiB uncompressed; per-file sizes and SHA-256 hashes in `DATA_MANIFEST.json`.

## Obtain the data

Clone this repository or use **Code → Download ZIP**. For the submission snapshot, download the source archive from the **v1.0.0 release**. The release includes the entire `data/` tree and its checksums; data are regular Git files, so Git LFS is not required.

## Contents and evidence

| Directory | Evidence |
|---|---|
| `data/results/benchmark/` | Saved predictive interfaces, sequential-query results, joint completions and checkpoint hash metadata |
| `data/results/identifiability/` | Two-world construction, regret lower bound and predictor audit summaries |
| `data/results/policy_controls/` | Matched policies, known-mechanism, oracle-coverage and calibration controls |
| `data/results/global_and_prediction/` | Global optimization, verified certificates, final-confirmation inference matrices and realized losses |
| `data/results/multiclass_and_calibration/` | Expanded pools, multiclass and calibration results |
| `data/results/sensitivity/` | Repair tolerance, calibration size and recorded local runtime results |
| `data/figure_data/` | Derived source tables retained from the original manuscript |
| `reproduction/code/` | Scientific implementations and saved-result validators |
| `reproduction/requirements/` | Separate requirements for cached validation and full inference |
| `reproduction/licenses/` | Third-party attribution and applicable supplied licence text |

`data/figure_sources.md` maps each final figure and table to its evidence. NPZ files contain NumPy arrays; CSV files contain result/source tables; JSON files contain seeds, configurations, indices, certificates and summaries. No model checkpoint is distributed. The manuscript, cover letter and editorial review files are not part of this data deposit.

## Validate recorded results

From the repository root, use Python 3.11 with `reproduction/requirements/checks.txt`:

```console
python -m pip install -r reproduction/requirements/checks.txt
python reproduction/validate.py --full
```

This recomputes checks from saved matrices without new pretrained-model inference. Full validation includes benchmark interfaces, joint models, the two-world construction, policy controls, final-confirmation losses and envelopes, and cached sensitivity results. Full model inference requires separately obtained third-party checkpoints; see `models/README.md` and `reproduction/requirements/inference.txt`.

## Provenance

Public tasks use the scikit-learn breast cancer, Iris, wine and digits loaders, restricted to the recorded binary tasks. Synthetic moons and mechanistic tasks are defined by the supplied generators and seeds. Upstream dataset records cited by the manuscript:

- [Breast Cancer Wisconsin (Diagnostic)](https://doi.org/10.24432/C5DW2B)
- [Iris](https://doi.org/10.24432/C56C76)
- [Wine](https://doi.org/10.24432/C5PC7J)
- [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49)

Published files are byte-identical to the local submission data at deposit time. `DATASET.json` and `CITATION.cff` record the creator, title, version and access location. No DOI has been assigned to this GitHub record.

## Rights and attribution

No blanket licence has been assigned to the author-generated data or custom code. Public access does not replace the applicable upstream data, software or model terms. The supplied Apache-2.0 text applies to the specifically attributed third-party figure-checking code; it is not a repository-wide licence. See `reproduction/licenses/NOTICE.md`.
