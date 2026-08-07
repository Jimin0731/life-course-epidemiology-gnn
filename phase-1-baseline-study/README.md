# Phase 1 — collaborative baseline study

This phase is collaborative work by **Alec Kemp, Jimin Byun, and Yuxin Du**. Jimin led practical implementation, experiment execution, production of results, and data analysis; the study remains group work and is not presented as Jimin's sole work.

## Research question

How do Logistic Regression, XGBoost, and ResNet behave as simulated cohort size and life-course exposure structure change?

## Paper-reported Phase 1 findings

For **Order1 at N=2,000**, the group paper reports approximately:

| Model | AUC | Interpretation in the paper |
|---|---:|---|
| XGBoost | 0.92 | Strongest of the three in this setting |
| ResNet | 0.69 | Sensitive to limited sample size, with overfitting/instability concerns |
| Logistic Regression | 0.50 | Did not capture the nonlinear and temporal pattern |

These are **paper-reported findings**, not reproduced repository metrics: the underlying Phase 1 result files were not uploaded. The broader conclusion is conditional, not universal—model ranking depends on cohort size and the rule structure. Some conclusions came from single experiment runs, so repeated random seeds are needed before making robustness claims.

## Layout

- `src/lce/` — importable baseline model and evaluation package.
- `scripts/data/` — R DAG simulation and life-course rule functions.
- `scripts/analysis/` — result parsing, aggregation, and plotting utilities.
- `scripts/run_baselines.sh` — baseline grid launcher.
- `historical/` — local wrappers retained for provenance, not the supported entry point.
- `results/` — reserved for regenerated outputs; no Phase 1 metric files were uploaded.

## Environment and run outline

```bash
conda env create -f environment.yml
conda activate lce-baseline
cd scripts/data
Rscript generate_simulated_data.R
cd ../..
PYTHONPATH=src python -m lce --data-dir scripts/data/data_500 \
  --output-dir results/model-results --data-name Order1 --model-name LR --epochs 1
```

The complete grid is computationally intensive and requires generated `.npy` data that is not committed.
