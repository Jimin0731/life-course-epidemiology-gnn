# Result provenance

## Canonical performance summary

The portfolio's sample-size and rule-family figures are derived from:

`phase-2-gnn-extension/results/raw/final_model_comparison_metrics_V4_N500_1000_2000.csv`

The file has 126 unique `(N, Rule, Model)` rows: 3 sample sizes × 14 rules × 3 model labels. Aggregation is an unweighted arithmetic mean across the 14 rule rows for each N/model or across rules within each rule family/model. No rows are filtered and no metric is imputed.

The portfolio display mapping follows the uploaded plotting script:

- `EconomicGNN_CNN` → PC-GNN
- `WeightedGAT_LSTM` → SPAN
- `SpatioTemporalGNN` → ILGNet

Both names are carried in machine-readable summaries.

## Node importance

The node-importance summary uses the uploaded `node_importance_summary_N500.csv`, `node_importance_summary_N1000.csv`, and `node_importance_summary_N2000.csv`. N=500 and N=2,000 each contain seven rules × nine nodes; N=1,000 contains two rules × nine nodes. Aggregates include observation and distinct-rule counts so the partial coverage is machine visible.

## Excluded from portfolio aggregation

Other result CSVs are retained but represent different model grids, sample-size grids, or experiment revisions. They are not merged because the upload does not provide a run registry establishing comparability. The empty `run_v3_n500.log` contains no evidence.
