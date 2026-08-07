# Scientific and reproducibility limitations

- The generated `.npy` datasets, trained checkpoints, and Phase 1 metric outputs were not uploaded. Full model reruns and baseline-vs-GNN numerical comparisons cannot be verified from this repository alone.
- The raw GNN CSVs do not include replicate IDs, fold-level predictions, confidence intervals, software versions, hardware metadata, or a single authoritative run registry.
- Several raw result files overlap but differ in model and sample-size grids. They are treated as separate revisions rather than pooled.
- The canonical V4 N=500/1,000/2,000 table appears complete as a grid, but each cell is a single reported value; uncertainty and run-to-run stability cannot be estimated.
- Optimal thresholds appear to be selected from validation outputs in experiment code. Without held-out predictions and checkpoints, threshold-selection leakage and calibration cannot be independently audited.
- Node-importance values are attention-derived and should not be interpreted as causal effects. N=1,000 coverage is partial (two rules), while N=500 and N=2,000 cover seven rules.
- Phase 1's R generator lists 12 rule functions while the Python CLI accepts 14 labels. Phase 2 contains both a 12-rule DAG exporter and a separate 14-rule generator whose function-to-paper-name mapping differs. These mismatches remain documented rather than silently harmonized.
- Environment files make dependencies explicit, but exact original package versions were absent; compatible version ranges are therefore a reconstruction, not a lockfile from the original runs.
- The portfolio names PC-GNN, SPAN, and ILGNet are mapped by the uploaded plotting script to legacy implementation labels. Raw evidence remains under legacy labels to avoid retrospective relabelling of results.
