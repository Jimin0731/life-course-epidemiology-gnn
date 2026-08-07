# Result provenance

This portfolio keeps three evidence classes separate. A label in every generated CSV and figure identifies which class applies.

## 1. Paper-reported results

The following files transcribe values supplied from the submitted papers; they are not substitutes for raw experiment rows:

- `results/paper-reported-overall-f1.csv` — overall mean F1 for ResNet, PC-GNN, SPAN, and ILGNet.
- `results/paper-reported-large-n-scaling.csv` — N=10,000/20,000 AUROC, F1, Brier, and reported F1 changes.
- `results/paper-reported-period4-f1.csv` — Period4 F1 values and differences against the ResNet 0.760 reference.
- Phase 1 README — Order1 at N=2,000 AUC values (XGBoost ≈0.92, ResNet ≈0.69, LR ≈0.50).

The rounded large-N GNN values agree with means recomputed from the uploaded `final_model_comparison_metrics_V4.csv` after the final 12-rule filter. They remain labelled paper-reported because that is the source of the portfolio's headline precision and interpretation. The ResNet values and Phase 1 metrics cannot be traced to uploaded metric files.

### Period4 internal inconsistency

The supplied paper values report PC-GNN at N=20,000 as F1 0.870 and difference versus ResNet +0.016, with ResNet F1 0.760. Direct arithmetic from those displayed values gives +0.110. The machine-readable table retains **both** the reported +0.016 and the recomputed +0.110, marks the row inconsistent, and does not choose a replacement. The qualitative statement that PC-GNN exceeds the reference at N=20,000 is supported by either positive difference.

## 2. Uploaded raw experiment results

Raw files under `phase-2-gnn-extension/results/raw/` remain unchanged after their provenance-preserving relocation. Two key revisions each contain all 14 exploratory rules, including `Period3` and `Timing4`:

| Raw revision | Grid | Role |
|---|---|---|
| `final_model_comparison_metrics_V4.csv` | 78 rows: 14 rules at N=10,000 and the final 12 at N=20,000, each × 3 labels | Large-N uploaded evidence |
| `final_model_comparison_metrics_V4_N500_1000_2000.csv` | 3 N × 14 rules × 3 labels = 126 rows | Earlier compact-sample uploaded evidence |

The raw labels are `EconomicGNN_CNN`, `WeightedGAT_LSTM`, and `SpatioTemporalGNN`. Paper model identities are not written back into these files. Other overlapping CSVs remain separate revisions because no run registry establishes that they can be pooled.

## 3. Portfolio-derived aggregates

The submitted individual paper uses exactly:

`Order1`, `Order2`, `Order3`, `Repeats1`, `Repeats2`, `Repeats3`, `Timing1`, `Timing2`, `Timing3`, `Period1`, `Period2`, `Period4`.

`Period3` was deliberately excluded; `Timing4` was not part of the final 12-task evaluation. The uploaded `plot_results.py` independently supports this scope by excluding both labels.

- `results/paper-aligned-large-n-raw-results.csv` is a deterministic 72-row view of `final_model_comparison_metrics_V4.csv`: 2 N × 12 rules × 3 raw labels. No metric is changed or imputed.
- `results/gnn-performance-by-rule-family.csv` averages that 72-row view within family and raw model label. The figure then averages equally over the three model-specific means. Each family contains three rules, both sample sizes, and all three labels (18 cells per displayed family mean).
- `results/gnn-performance-by-sample-size.csv` retains the earlier compact revision as exploratory evidence, filtered to the same 12 rules (12 cells per N/label).
- `results/node-importance-by-sample-size.csv` averages uploaded `WeightedGAT_LSTM` attention-derived values after the final-rule filter. N=1,000 contains two rules; N=500 and N=2,000 contain seven available rules.

Aggregates are unweighted arithmetic means. Available files contain single cell estimates, not replicate IDs, fold-level predictions, confidence intervals, or standard errors.

## Figure evidence labels

| Figure | Evidence source |
|---|---|
| `assets/research-evolution.svg` | Schematic research narrative |
| `assets/period4-f1.svg` | Paper-reported result |
| `assets/large-n-scaling.svg` | Paper-reported result |
| `assets/performance-by-rule-family.svg` | Derived from uploaded large-N raw CSV, final 12-rule filter |
| `assets/performance-across-sample-sizes.svg` | Exploratory aggregate from uploaded compact raw CSV, final 12-rule filter |
| `assets/span-gnn-architecture.svg` | Source-code-based schematic |
| `assets/node-importance.svg` | Derived from uploaded attention summaries; coverage caveats apply |

Display names in raw-derived assets follow the original plotting script and are explicitly described as display mappings. See `reports/model-identity-audit.md` before treating them as architecture identities.
