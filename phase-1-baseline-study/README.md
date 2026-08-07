# Phase 1 — collaborative baseline study

This phase is the collaborative work of **Alec Kemp, Jimin Byun, and Yuxin Du**. Jimin led implementation, experiment execution, result generation, and analysis; the study remains group work and is not presented as Jimin's sole work.

## Research question

How do Logistic Regression, XGBoost, and ResNet behave as the simulated cohort size changes across distinct life-course exposure rules?

The simulation defines nine variables over 40 time points. The generator enumerates N=500, 1,000, 2,000, 5,000, 10,000, and 20,000. The model pipeline evaluates discrimination, classification, calibration, and runtime metrics.

## Layout

- `src/lce/` — importable baseline model and evaluation package.
- `scripts/data/` — R DAG simulation and life-course rule functions.
- `scripts/analysis/` — result parsing, aggregation, and plotting utilities.
- `scripts/run_baselines.sh` — baseline grid launcher.
- `historical/` — local wrappers retained for provenance but not treated as the supported entry point.
- `results/` — reserved for regenerated Phase 1 outputs. No Phase 1 result files were present in the upload.

## Environment and run outline

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate lce-baseline
```

Generate data from `scripts/data/` so that the R `source()` call resolves, then expose the package and run one model/rule combination:

```bash
cd scripts/data
Rscript generate_simulated_data.R
cd ../..
PYTHONPATH=src python -m lce --data-dir scripts/data/data_500 --output-dir results/model-results \
  --data-name Order1 --model-name LR --epochs 1
```

The complete grid is computationally intensive. The generated `.npy` data and Phase 1 model outputs were not uploaded, so this portfolio does not display baseline metric values.
