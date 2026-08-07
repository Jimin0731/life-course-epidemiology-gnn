# Model identity audit

## Conclusion

The repository supports the original plotting script's labels as a **portfolio display mapping**, but it does not support treating all three as conclusively registered paper-to-class identities.

In particular, the ILGNet implementation identity is unresolved.

| Paper/display name | Plot-script source label | Confidence | Conclusion |
|---|---|---|---|
| PC-GNN | `EconomicGNN_CNN` | Moderate | Architecture is broadly consistent, but there is no independent paper-name registry in the runner or raw CSV |
| SPAN | `WeightedGAT_LSTM` | High | LSTM + edge-aware GATv2 + returned attention closely matches the paper description |
| ILGNet | `SpatioTemporalGNN` | Low / unresolved | Plotting evidence conflicts with architectural evidence; do not silently relabel raw results |

## Trace performed

### Plotting and result labels

`phase-2-gnn-extension/scripts/analysis/plot_results.py` contains:

- `economicgnn_cnn` → `PC-GNN`
- `weightedgat_lstm` → `SPAN`
- `spatiotemporalgnn` → `ILGNet`

The active runner imports `EconomicGNN`, `WeightedGATGNN`, and `SpatioTemporalGNN`, then writes the corresponding raw labels `EconomicGNN_CNN`, `WeightedGAT_LSTM`, and `SpatioTemporalGNN`. The V4 raw tables use those labels. The runner does not import or execute `HybridGNN` in its active three-model grid.

Historical temporal runners include `HybridGNN_LSTM` alongside other variants. Earlier raw revisions contain that label, but no committed run registry connects those rows to a particular submitted-paper table.

### PC-GNN expectation versus `EconomicGNN`

The temporal `EconomicGNN` selects a temporal encoder, then uses two GCN layers when edge weights are enabled or two GAT layers otherwise. Both graph stages use LayerNorm, followed by a small feed-forward head. The plotted `EconomicGNN_CNN` variant is therefore lightweight relative to the other uploaded models and broadly matches the paper's temporal + GCN/GAT + LayerNorm description.

This is architectural consistency, not definitive identity: `PC-GNN` is not the class name, runner label, or raw result label.

### SPAN expectation versus `WeightedGATGNN`

`WeightedGATGNN` supports an LSTM temporal encoder, uses two `GATv2Conv` layers with edge weights passed as one-dimensional edge attributes, and returns both layers' attention weights. `WeightedGAT_LSTM` therefore strongly matches the paper's SPAN description and the attention-analysis workflow.

The raw name remains unchanged. Figures say SPAN is the plot-script display label for `WeightedGAT_LSTM`.

### ILGNet expectation versus the two candidate classes

The paper describes a dual-branch architecture: a GCN branch, a GAT branch, concatenation, and a fusion layer.

`hybridgnn_temporal.py` implements exactly those structural elements after temporal encoding:

1. input projection;
2. two-layer GCN branch;
3. parallel GAT branch;
4. `torch.cat([h_gcn, h_gat])`;
5. learned fusion layer.

By contrast, `spatiotemporal.py` implements temporal self-attention, temporal↔graph cross-attention, GATv2 message passing, and temporal/graph fusion. That is a meaningful architecture, but it does not obviously match the paper's stated ILGNet dual GCN/GAT design.

The available evidence therefore conflicts:

- **plot and final V4 label evidence:** `SpatioTemporalGNN` was displayed as ILGNet;
- **architectural evidence:** `HybridGNN` is the closer paper-description match;
- **runner evidence:** the active final runner uses `SpatioTemporalGNN`, while historical runners include `HybridGNN_LSTM`;
- **missing evidence:** no checkpoint manifest, run registry, paper appendix mapping, or commit tag identifies which implementation produced each named paper model.

## Portfolio policy

- Raw labels and class names remain unchanged.
- Paper-reported tables use paper names because their source is the paper, not a raw-label transformation.
- Raw-derived summaries retain both the source model and display label plus an identity-status field.
- No architectural claim depends on the ambiguous `SpatioTemporalGNN` → ILGNet mapping.
- Resolving ILGNet requires an authoritative experiment registry or paper-to-code mapping from the original research workflow.
