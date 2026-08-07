# Original upload preservation and archive map

The rebuild began from `main` at `06a9c193d85b655bb82a9224bbd1086777dd9755`. Git history is the byte-for-byte archive; `original-file-manifest.sha256` records all 67 uploaded paths and digests from that commit.

| Original area | New area |
|---|---|
| `Base-project/` | `phase-1-baseline-study/` |
| `Developed/` | `phase-2-gnn-extension/` |
| Phase 1 importable Python | `phase-1-baseline-study/src/lce/` |
| Phase 1 data/analysis launchers | `phase-1-baseline-study/scripts/` |
| Phase 2 importable Python | `phase-2-gnn-extension/src/lce_gnn/` |
| Phase 2 experiment/data/analysis launchers | `phase-2-gnn-extension/scripts/` |
| Uploaded Phase 2 CSV/log outputs | `phase-2-gnn-extension/results/raw/` |
| Ambiguous, local, or superseded prototypes | Each phase's `historical/` directory |

Machine-specific absolute paths in four historical helpers were replaced with relative placeholders. That sanitation is deliberate; use the source commit plus manifest to inspect exact originals.
