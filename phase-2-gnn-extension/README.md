# Phase 2 — Jimin Byun's individual GNN extension

This individual project asks whether explicitly using the simulated life-course DAG can improve structural alignment, scaling, or descriptive interpretability beyond the Phase 1 baselines. ResNet remains strongest overall; the contribution is task-dependent GNN behavior rather than universal superiority.

## Final paper scope

The submitted paper evaluates exactly 12 binary tasks:

- Order1, Order2, Order3
- Repeats1, Repeats2, Repeats3
- Timing1, Timing2, Timing3
- Period1, Period2, Period4

`Period3` was deliberately excluded and `Timing4` was not part of the final evaluation. Both remain unchanged in earlier 14-rule raw CSVs for provenance. Paper-facing summaries filter them out.

## Key findings

**Paper-reported overall mean F1:** ResNet 0.954, PC-GNN 0.809, SPAN 0.679, ILGNet 0.649. ResNet therefore leads overall.

| Paper model | N=10,000 AUROC / F1 / Brier | N=20,000 AUROC / F1 / Brier | F1 change |
|---|---|---|---:|
| PC-GNN | 0.976 / 0.784 / 0.037 | 0.988 / 0.835 / 0.035 | +0.051 |
| SPAN | 0.886 / 0.686 / 0.105 | 0.882 / 0.672 / 0.108 | -0.015 |
| ILGNet | 0.828 / 0.615 / 0.185 | 0.898 / 0.684 / 0.130 | **+0.069** |

ILGNet shows the strongest paper-reported data-scaling effect. Across rule families, the reported/descriptive pattern is **Period > Repeats > Timing > Order**, consistent with—but not proof of—better alignment between some data-generating mechanisms and DAG message passing.

### Period4

With the paper's ResNet reference F1 of 0.760, SPAN and ILGNet outperform the reference at both N=10,000 and N=20,000. PC-GNN exceeds it at N=20,000 but not N=10,000. The maximum paper-reported improvement is +0.149 for ILGNet at N=10,000.

The paper-provided PC-GNN N=20,000 values contain an internal arithmetic discrepancy (F1 0.870, reported difference +0.016). Both are retained verbatim and flagged in the provenance report.

## Model identity boundary

The uploaded plotting script used this **portfolio display mapping**:

| Paper/display name | Uploaded label in that script | Audit conclusion |
|---|---|---|
| PC-GNN | `EconomicGNN_CNN` | Broad architectural support: temporal encoder, GCN/GAT path, LayerNorm; exact paper identity not independently registered |
| SPAN | `WeightedGAT_LSTM` | Strong support: LSTM + edge-aware GATv2, with attention retained |
| ILGNet | `SpatioTemporalGNN` | **Unresolved:** this class uses temporal self-/cross-attention and GATv2, while `HybridGNN` more closely matches the paper's dual GCN/GAT branch and fusion description |

No class or raw result label has been silently renamed. See [`../reports/model-identity-audit.md`](../reports/model-identity-audit.md).

![SPAN architecture schematic](../assets/span-gnn-architecture.svg)

## Layout and evidence

- `src/lce_gnn/` — active model package; historical paper identity is not implied by class location.
- `scripts/experiments/` — active temporal experiment runner.
- `scripts/analysis/` — plotting, hyperparameter, and attention-analysis tools.
- `scripts/data/` — DAG/data generators whose 12- vs 14-rule differences remain documented.
- `config/` — N-adaptive configuration.
- `results/raw/` — uploaded result files unchanged after relocation.
- `historical/` — traced prototypes, retained but not treated as active.

The canonical paper-facing row layer is [`../results/paper-aligned-large-n-raw-results.csv`](../results/paper-aligned-large-n-raw-results.csv): the uploaded N=10,000/20,000 V4 table filtered to the final 12 rules. Paper-reported summaries are separate files; compact N=500/1,000/2,000 results remain exploratory raw-derived evidence.

## Environment and run outline

```bash
conda env create -f environment.yml
conda activate lce-gnn
PYTHONPATH=src python scripts/experiments/run_temporal_experiments.py
```

Full execution requires generated DAG/data files and is not exercised in CI. CI compiles code, regenerates evidence deterministically, verifies provenance, and tests the runner's prevalence ordering without importing heavy ML dependencies.
