# Machine-readable result layers

Generated deterministically by `python3 scripts/build_portfolio_assets.py`.

## Paper-reported

- `paper-reported-overall-f1.csv`
- `paper-reported-large-n-scaling.csv`
- `paper-reported-period4-f1.csv`

These transcribe paper values and carry `evidence_type=paper-reported-result`. The Period4 file preserves and flags the PC-GNN N=20,000 arithmetic inconsistency.

## Final-paper scope

- `final-paper-rule-set.csv` — exact 12-rule definition.
- `paper-aligned-large-n-raw-results.csv` — 72 unchanged metric rows selected from the uploaded large-N 14-rule V4 CSV (2 N × 12 rules × 3 raw labels).

## Portfolio-derived

- `gnn-performance-by-rule-family.csv` — final-12-rule large-N aggregates.
- `gnn-performance-by-sample-size.csv` — final-12-rule compact N=500/1,000/2,000 exploratory aggregates.
- `node-importance-by-sample-size.csv` — attention-derived summary with coverage fields.
- `result-catalog.csv` — inventory of preserved raw files.

Raw files remain under `phase-2-gnn-extension/results/raw/`. Read `reports/result-provenance.md` and `reports/model-identity-audit.md` before joining or relabelling layers.
