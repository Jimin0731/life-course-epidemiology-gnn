# Ambiguous filename audit

| Original path | Finding from content/import trace | Portfolio path and action |
|---|---|---|
| `Developed/Untitled.R` | Standalone generator for five N values and 14 rule outputs; not sourced by another file | `scripts/data/generate_all_14_rules_legacy_mapping.R`; retained descriptively alongside, but not merged with, the 12-rule DAG exporter |
| `Developed/__init__1.py` | Alternate package export file with the same temporal-model exports plus comments; never imported | `historical/source/package_init_temporal_legacy.py`; archived |
| `Developed/Run_all_n_temporal1.py` | Earlier five-N, five-model F1-threshold experiment runner | `historical/scripts/run_temporal_experiments_f1_legacy.py`; archived |
| `Developed/temporal_encoders1.py` | The encoder module actually imported by `__init__.py` and all three temporal graph-model modules | `src/lce_gnn/temporal_encoders.py`; promoted and all imports updated |
| `Developed/temporal_encoders.py` | Larger experimental encoder revision, but not imported by the uploaded package | `historical/source/temporal_encoders_experimental.py`; archived |
| `Developed/run_all_N(simple).py` | Early self-contained graph runner with a shell-unfriendly filename | `historical/scripts/run_graph_experiments_simple_legacy.py`; archived |

`Run_all_n_temporal.py` is the later feature-engineered/adaptive runner and is now `scripts/experiments/run_temporal_experiments.py`. The active script imports the cleaned `lce_gnn` package instead of unresolved `Data_simulation.models` paths.
