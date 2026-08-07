# Scientific and reproducibility limitations

- The study uses synthetic longitudinal data. Performance does not establish clinical utility, transportability, causal discovery, or performance on observed epidemiological cohorts.
- Generated `.npy` datasets and trained checkpoints were not uploaded, so full experiment reruns and prediction-level audits cannot be completed from this repository alone.
- Phase 1 metric outputs were not uploaded. Its Order1 findings are paper-reported and cannot be independently recomputed here.
- Raw GNN tables do not contain replicate IDs, fold-level predictions, random seeds, confidence intervals, or a definitive run registry. Cell values appear to be single-run estimates, so uncertainty and run-to-run stability cannot be estimated.
- Overlapping raw result files differ in sample/model grids and are treated as separate revisions rather than pooled.
- The final paper uses 12 tasks, while preserved exploratory CSVs include `Period3` and `Timing4`. Paper-facing views filter those two without changing raw evidence.
- One supplied Period4 comparison is internally inconsistent: PC-GNN N=20,000 F1 0.870 against ResNet 0.760 is paired with reported difference +0.016. Both the reported and arithmetically implied values are retained for review.
- The original plotting script's ILGNet display mapping points to `SpatioTemporalGNN`, but the paper's dual GCN/GAT branch description more closely matches `HybridGNN`. No experiment registry resolves this identity; see `model-identity-audit.md`.
- Attention-derived importance is descriptive/correlational. It does not establish a causal effect or prove that the model recovered the data-generating DAG. Stronger interpretation would require independent methods such as perturbation tests, SHAP, or attribution comparisons.
- N=1,000 node-importance coverage is partial (two rules); N=500 and N=2,000 cover seven uploaded rules, not all final 12.
- Optimal thresholds appear to be selected from validation outputs. Without held-out predictions and checkpoints, threshold-selection leakage and calibration cannot be independently audited.
- The R generators and accepted CLI labels have historical 12- versus 14-rule mapping differences. They remain documented instead of being silently harmonized.
- Reconstructed environment files use compatible constraints because exact package versions and hardware metadata from the original runs were absent.
