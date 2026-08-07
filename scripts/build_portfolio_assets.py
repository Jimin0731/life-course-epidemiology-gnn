#!/usr/bin/env python3
"""Build deterministic, dependency-free paper-aligned CSV and SVG artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "phase-2-gnn-extension" / "results" / "raw"
RAW_COMPACT = RAW / "final_model_comparison_metrics_V4_N500_1000_2000.csv"
RAW_LARGE = RAW / "final_model_comparison_metrics_V4.csv"

FINAL_RULES = (
    "Order1", "Order2", "Order3",
    "Repeats1", "Repeats2", "Repeats3",
    "Timing1", "Timing2", "Timing3",
    "Period1", "Period2", "Period4",
)
EXPERIMENTAL_ONLY_RULES = {"Period3", "Timing4"}
RAW_MODELS = ("EconomicGNN_CNN", "WeightedGAT_LSTM", "SpatioTemporalGNN")
DISPLAY_NAMES = {
    "EconomicGNN_CNN": "PC-GNN",
    "WeightedGAT_LSTM": "SPAN",
    "SpatioTemporalGNN": "ILGNet",
}
IDENTITY_STATUS = {
    "EconomicGNN_CNN": "plot-script mapping; architecture broadly consistent",
    "WeightedGAT_LSTM": "plot-script mapping; architecture strongly supported",
    "SpatioTemporalGNN": "plot-script mapping only; architectural identity unresolved",
}
COLORS = {
    "EconomicGNN_CNN": "#1967d2",
    "WeightedGAT_LSTM": "#00897b",
    "SpatioTemporalGNN": "#e76f51",
    "ResNet": "#6d597a",
}
NODE_ORDER = (
    "sex", "ethnicity", "marriage", "education", "class",
    "admissions", "admissions2", "admissions3", "admissions4",
)

PAPER_OVERALL = (
    ("ResNet", 0.954), ("PC-GNN", 0.809),
    ("SPAN", 0.679), ("ILGNet", 0.649),
)
PAPER_SCALING = (
    ("PC-GNN", 10000, 0.976, 0.784, 0.037, ""),
    ("PC-GNN", 20000, 0.988, 0.835, 0.035, "+0.051"),
    ("SPAN", 10000, 0.886, 0.686, 0.105, ""),
    ("SPAN", 20000, 0.882, 0.672, 0.108, "-0.015"),
    ("ILGNet", 10000, 0.828, 0.615, 0.185, ""),
    ("ILGNet", 20000, 0.898, 0.684, 0.130, "+0.069"),
)
PAPER_PERIOD4 = (
    ("PC-GNN", 10000, 0.743, -0.017),
    ("PC-GNN", 20000, 0.870, 0.016),
    ("SPAN", 10000, 0.905, 0.145),
    ("SPAN", 20000, 0.851, 0.091),
    ("ILGNet", 10000, 0.909, 0.149),
    ("ILGNet", 20000, 0.901, 0.141),
)
RESNET_PERIOD4 = 0.760


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def validate_raw(rows: list[dict[str, str]], sample_sizes: set[int]) -> None:
    required = {"N", "Rule", "Model", "AUROC", "F1_Score", "Brier_Score"}
    if not rows or required - set(rows[0]):
        raise ValueError("Raw result file is empty or missing required columns")
    keys = {(row["N"], row["Rule"], row["Model"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Expected unique N × Rule × Model rows")
    if {int(row["N"]) for row in rows} != sample_sizes:
        raise ValueError("Unexpected sample-size grid")
    if {row["Model"] for row in rows} != set(RAW_MODELS):
        raise ValueError("Unexpected model grid")
    expected_final = {
        (str(n_value), rule, model)
        for n_value in sample_sizes for rule in FINAL_RULES for model in RAW_MODELS
    }
    if not expected_final <= keys:
        raise ValueError("Final 12-rule grid is incomplete")
    rules = {row["Rule"] for row in rows}
    if len(rules) != 14 or not EXPERIMENTAL_ONLY_RULES <= rules:
        raise ValueError("Expected preserved 14-rule evidence including Period3 and Timing4")


def paper_aligned_rows(rows: list[dict[str, str]], source: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if row["Rule"] not in FINAL_RULES:
            continue
        output.append({
            **row,
            "evidence_type": "uploaded-raw-result",
            "paper_rule_scope": "final-12",
            "source_file": source.name,
        })
    return output


def aggregate(group: list[dict[str, str]]) -> dict[str, object]:
    return {
        "observations": len(group),
        "mean_auroc": f"{fmean(float(row['AUROC']) for row in group):.6f}",
        "mean_f1_score": f"{fmean(float(row['F1_Score']) for row in group):.6f}",
        "mean_brier_score": f"{fmean(float(row['Brier_Score']) for row in group):.6f}",
    }


def performance_by_sample(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["Rule"] in FINAL_RULES:
            groups[(int(row["N"]), row["Model"])].append(row)
    output = []
    for n_value in (500, 1000, 2000):
        for model in RAW_MODELS:
            output.append({
                "sample_size": n_value,
                "display_label": DISPLAY_NAMES[model],
                "source_model": model,
                "display_mapping_status": IDENTITY_STATUS[model],
                **aggregate(groups[(n_value, model)]),
                "rule_scope": "final-paper-12",
                "excluded_raw_rules": "Period3|Timing4",
                "source_file": RAW_COMPACT.name,
            })
    return output


def performance_by_family(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["Rule"] not in FINAL_RULES:
            continue
        family = "Repeats" if row["Rule"].startswith("Repeats") else row["Rule"].rstrip("1234")
        groups[(family, row["Model"])].append(row)
    output = []
    for family in ("Period", "Repeats", "Timing", "Order"):
        for model in RAW_MODELS:
            output.append({
                "rule_family": family,
                "display_label": DISPLAY_NAMES[model],
                "source_model": model,
                "display_mapping_status": IDENTITY_STATUS[model],
                **aggregate(groups[(family, model)]),
                "sample_sizes": "10000|20000",
                "rule_scope": "final-paper-12",
                "source_file": RAW_LARGE.name,
            })
    return output


def paper_rule_rows() -> list[dict[str, object]]:
    return [{
        "rule": rule,
        "rule_family": "Repeats" if rule.startswith("Repeats") else rule.rstrip("1234"),
        "included_in_final_paper": "true",
        "evidence_type": "paper-definition",
    } for rule in FINAL_RULES]


def paper_overall_rows() -> list[dict[str, object]]:
    return [{"model": model, "overall_mean_f1": f"{f1:.3f}", "evidence_type": "paper-reported-result"}
            for model, f1 in PAPER_OVERALL]


def paper_scaling_rows() -> list[dict[str, object]]:
    return [{
        "model": model, "sample_size": n_value, "auroc": f"{auroc:.3f}",
        "f1": f"{f1:.3f}", "brier": f"{brier:.3f}",
        "f1_change_from_n10000": change, "evidence_type": "paper-reported-result",
    } for model, n_value, auroc, f1, brier, change in PAPER_SCALING]


def paper_period4_rows() -> list[dict[str, object]]:
    output = []
    for model, n_value, f1, reported_diff in PAPER_PERIOD4:
        computed = round(f1 - RESNET_PERIOD4, 3)
        consistent = abs(computed - reported_diff) < 0.0005
        output.append({
            "model": model,
            "sample_size": n_value,
            "f1": f"{f1:.3f}",
            "resnet_reference_f1": f"{RESNET_PERIOD4:.3f}",
            "paper_reported_difference_vs_resnet": f"{reported_diff:+.3f}",
            "arithmetic_difference_from_displayed_values": f"{computed:+.3f}",
            "internal_consistency": "consistent" if consistent else "paper-values-conflict-retained-verbatim",
            "evidence_type": "paper-reported-result",
        })
    return output


def node_importance_summary() -> list[dict[str, object]]:
    sources = {n: RAW / f"node_importance_summary_N{n}.csv" for n in (500, 1000, 2000)}
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for n_value, source in sources.items():
        for row in read_csv(source):
            if row["Rule"] in FINAL_RULES:
                groups[(n_value, row["Node"])].append(row)
    output = []
    for n_value in (500, 1000, 2000):
        for node in NODE_ORDER:
            group = groups[(n_value, node)]
            output.append({
                "sample_size": n_value, "node": node,
                "mean_importance": f"{fmean(float(row['Importance']) for row in group):.6f}",
                "observations": len(group), "rule_count": len({row["Rule"] for row in group}),
                "coverage": "partial" if n_value == 1000 else "available-set-complete",
                "interpretation": "descriptive-attention-not-causal", "source_file": sources[n_value].name,
            })
    return output


def result_catalog() -> list[dict[str, object]]:
    output = []
    for path in sorted(RAW.iterdir(), key=lambda item: item.name.lower()):
        if path.name == "README.md":
            continue
        if path.suffix.lower() != ".csv":
            output.append({"file": path.name, "kind": "log", "rows": 0, "sample_size_count": 0,
                           "rule_count": 0, "model_count": 0, "node_count": 0,
                           "status": "empty" if path.stat().st_size == 0 else "present"})
            continue
        rows = read_csv(path)
        columns = set(rows[0]) if rows else set()
        output.append({
            "file": path.name,
            "kind": "node_importance" if "Importance" in columns else "model_performance",
            "rows": len(rows),
            "sample_size_count": len({row.get("N", "") for row in rows if row.get("N", "")}),
            "rule_count": len({row.get("Rule", "") for row in rows if row.get("Rule", "")}),
            "model_count": len({row.get("Model", "") for row in rows if row.get("Model", "")}),
            "node_count": len({row.get("Node", "") for row in rows if row.get("Node", "")}),
            "status": "preserved-raw-14-rule-source" if path in {RAW_COMPACT, RAW_LARGE} else "retained-revision",
        })
    return output


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{title}</title>', f'<desc id="desc">{subtitle}</desc>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#183153}.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#526579}.label{font-size:14px}.small{font-size:12px;fill:#526579}.value{font-size:12px;font-weight:700}.box{stroke:#c9d5e3;stroke-width:1.5}.axis{stroke:#9eafc1}.grid{stroke:#e5ebf2}.arrow{stroke:#49647e;stroke-width:2;fill:none}</style>',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        f'<text x="50" y="50" class="title">{title}</text>',
        f'<text x="50" y="76" class="sub">{subtitle}</text>',
    ]


def research_evolution_svg() -> str:
    p = svg_start(1240, 420, "Research evolution", "Baseline evidence motivated a DAG-aware individual extension")
    p += [
        '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#49647e"/></marker></defs>',
        '<rect x="45" y="125" width="330" height="205" rx="18" fill="#e8f1ff" class="box"/>',
        '<text x="70" y="160" font-size="14" font-weight="700" fill="#1967d2">PHASE 1 · COLLABORATIVE</text>',
        '<text x="70" y="198" font-size="21" font-weight="700">Baseline comparison</text>',
        '<text x="70" y="232" class="label">LR · XGBoost · ResNet</text>',
        '<text x="70" y="263" class="small">Cohort size and rule structure</text>',
        '<text x="70" y="286" class="small">changed relative performance.</text>',
        '<path d="M385 228 L430 228" class="arrow" marker-end="url(#a)"/>',
        '<rect x="445" y="125" width="350" height="205" rx="18" fill="#fff6e4" class="box"/>',
        '<text x="470" y="160" font-size="14" font-weight="700" fill="#9a6700">RESEARCH MOTIVATION</text>',
        '<text x="470" y="198" font-size="21" font-weight="700">Limitations identified</text>',
        '<text x="470" y="232" class="label">Small-N instability</text>',
        '<text x="470" y="260" class="label">DAG structure not explicit</text>',
        '<text x="470" y="288" class="small">Deep learning was not uniformly superior.</text>',
        '<path d="M805 228 L850 228" class="arrow" marker-end="url(#a)"/>',
        '<rect x="865" y="125" width="330" height="205" rx="18" fill="#e6f7f3" class="box"/>',
        '<text x="890" y="160" font-size="14" font-weight="700" fill="#00897b">PHASE 2 · INDIVIDUAL</text>',
        '<text x="890" y="198" font-size="21" font-weight="700">DAG-aware GNNs</text>',
        '<text x="890" y="232" class="label">PC-GNN · SPAN · ILGNet</text>',
        '<text x="890" y="263" class="small">Structural alignment, scaling,</text>',
        '<text x="890" y="286" class="small">and descriptive attention.</text>',
        '<text x="45" y="385" class="small">SCHEMATIC — Phase 1: Alec Kemp, Jimin Byun, Yuxin Du. Phase 2: Jimin Byun.</text>',
        '</svg>\n',
    ]
    return "\n".join(p)


def line_chart(title: str, subtitle: str, values: dict[str, tuple[float, float]], footer: str) -> str:
    p = svg_start(1000, 590, title, subtitle)
    left, top, bottom = 110, 125, 465
    def y(v: float) -> float: return bottom - (v - 0.55) / 0.40 * (bottom - top)
    for tick in (0.6, 0.7, 0.8, 0.9):
        yy = y(tick); p += [f'<line x1="{left}" y1="{yy:.1f}" x2="900" y2="{yy:.1f}" class="grid"/>', f'<text x="70" y="{yy+5:.1f}" class="small">{tick:.1f}</text>']
    p += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>', f'<line x1="{left}" y1="{bottom}" x2="900" y2="{bottom}" class="axis"/>', '<text x="280" y="505" class="label">N=10,000</text>', '<text x="700" y="505" class="label">N=20,000</text>', '<text x="24" y="320" class="label" transform="rotate(-90 24 320)">F1 score</text>']
    raw_for_name = {v: k for k, v in DISPLAY_NAMES.items()}
    for index, (name, pair) in enumerate(values.items()):
        color = COLORS[raw_for_name[name]]
        y1, y2 = y(pair[0]), y(pair[1])
        p += [f'<line x1="315" y1="{y1:.1f}" x2="735" y2="{y2:.1f}" stroke="{color}" stroke-width="4"/>', f'<circle cx="315" cy="{y1:.1f}" r="7" fill="{color}"/>', f'<circle cx="735" cy="{y2:.1f}" r="7" fill="{color}"/>', f'<text x="280" y="{y1-12:.1f}" class="value">{pair[0]:.3f}</text>', f'<text x="745" y="{y2+5:.1f}" class="value">{pair[1]:.3f}</text>', f'<rect x="{610 + (index%2)*170}" y="{92 + (index//2)*24}" width="14" height="14" fill="{color}"/>', f'<text x="{632 + (index%2)*170}" y="{104 + (index//2)*24}" class="small">{name}</text>']
    p += [f'<text x="110" y="555" class="small">{footer}</text>', '</svg>\n']
    return "\n".join(p)


def period4_svg() -> str:
    p = svg_start(1060, 640, "Period4 F1 against the ResNet reference", "Paper-reported result; differences are retained exactly as reported")
    left, top, bottom = 105, 130, 500
    def y(v: float) -> float: return bottom - (v - 0.70) / 0.25 * (bottom - top)
    for tick in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        yy=y(tick); p += [f'<line x1="{left}" y1="{yy:.1f}" x2="1000" y2="{yy:.1f}" class="grid"/>', f'<text x="65" y="{yy+5:.1f}" class="small">{tick:.2f}</text>']
    ref_y=y(RESNET_PERIOD4); p += [f'<line x1="{left}" y1="{ref_y:.1f}" x2="1000" y2="{ref_y:.1f}" stroke="{COLORS["ResNet"]}" stroke-width="3" stroke-dasharray="8 5"/>', f'<text x="112" y="{ref_y-8:.1f}" class="small">ResNet reference 0.760</text>']
    data={(m,n):(f,d) for m,n,f,d in PAPER_PERIOD4}; names=("PC-GNN","SPAN","ILGNet"); raw_for_name={v:k for k,v in DISPLAY_NAMES.items()}
    for group,(n,cx) in enumerate(((10000,325),(20000,745))):
        p.append(f'<text x="{cx-35}" y="565" class="label">N={n:,}</text>')
        for idx,name in enumerate(names):
            f1,diff=data[(name,n)]; x=cx+(idx-1)*90; yy=y(f1); color=COLORS[raw_for_name[name]]
            p += [f'<rect x="{x-26}" y="{yy:.1f}" width="52" height="{bottom-yy:.1f}" rx="4" fill="{color}"/>', f'<text x="{x-22}" y="{yy-9:.1f}" class="value">{f1:.3f}</text>', f'<text x="{x-24}" y="{bottom+18}" class="small">{name}</text>', f'<text x="{x-23}" y="{bottom+37}" class="small">{diff:+.3f}</text>']
    p += ['<text x="105" y="595" class="small">SPAN and ILGNet exceed ResNet at both sizes; PC-GNN does so only at N=20,000.</text>', '<text x="105" y="617" class="small">Caveat: PC-GNN N=20k reported +0.016 differs from arithmetic +0.110; both are retained in provenance.</text>', '</svg>\n']
    return "\n".join(p)


def family_svg(rows: list[dict[str, object]]) -> str:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows: groups[str(row["rule_family"])].extend([float(row["mean_f1_score"])] * int(row["observations"]))
    # Each model-specific mean represents the same number of cells, so an unweighted mean is equivalent.
    values={family:fmean(float(r["mean_f1_score"]) for r in rows if r["rule_family"]==family) for family in ("Period","Repeats","Timing","Order")}
    p=svg_start(920,600,"Performance by final 12-rule family","Portfolio-derived mean F1 across three uploaded model labels and N=10,000/20,000")
    left,top,bottom=100,125,470
    def y(v:float)->float:return bottom-(v-0.45)/0.55*(bottom-top)
    for tick in (0.5,0.6,0.7,0.8,0.9,1.0):
        yy=y(tick);p += [f'<line x1="{left}" y1="{yy:.1f}" x2="860" y2="{yy:.1f}" class="grid"/>',f'<text x="60" y="{yy+5:.1f}" class="small">{tick:.1f}</text>']
    colors=("#5e60ce","#4ea8de","#56c596","#f4a261")
    for family,x,color in zip(("Period","Repeats","Timing","Order"),(190,390,590,790),colors):
        v=values[family];yy=y(v);p += [f'<rect x="{x-42}" y="{yy:.1f}" width="84" height="{bottom-yy:.1f}" rx="6" fill="{color}"/>',f'<text x="{x-28}" y="{yy-10:.1f}" class="value">{v:.4f}</text>',f'<text x="{x-28}" y="500" class="label">{family}</text>']
    p += ['<text x="100" y="548" class="small">Source: final_model_comparison_metrics_V4.csv filtered to the final 12 rules; descriptive aggregate, no uncertainty.</text>', '<text x="100" y="572" class="small">Pattern: Period &gt; Repeats &gt; Timing &gt; Order. This is consistent with structural alignment; it is not causal proof.</text>', '</svg>\n']
    return "\n".join(p)


def compact_svg(rows: list[dict[str, object]]) -> str:
    values={(int(r["sample_size"]),str(r["source_model"])):float(r["mean_auroc"]) for r in rows}
    pairs={DISPLAY_NAMES[m]:(values[(500,m)],values[(2000,m)]) for m in RAW_MODELS}
    return line_chart("Exploratory compact-sample performance","Mean AUROC endpoints from the uploaded compact revision, final 12-rule filter",pairs,"DERIVED FROM UPLOADED RAW CSV — N=1,000 remains in the machine-readable summary; model names are plot-script display labels.")


def architecture_svg() -> str:
    p=svg_start(1120,520,"SPAN architecture","SCHEMATIC of the uploaded WeightedGAT_LSTM implementation; not a measured result")
    boxes=((45,155,190,"Longitudinal nodes","9 variables × 40 times"),(300,155,190,"LSTM encoder","Per-node temporal history"),(555,135,220,"Weighted GATv2","Edge-aware attention retained"),(845,155,220,"Prediction / importance","Descriptive, not causal"))
    p.append('<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#49647e"/></marker></defs>')
    for i,(x,y,w,title,sub) in enumerate(boxes):
        p += [f'<rect x="{x}" y="{y}" width="{w}" height="150" rx="15" fill="{("#e8f1ff","#eef3f8","#e6f7f3","#f7f0ff")[i]}" class="box"/>',f'<text x="{x+22}" y="{y+55}" font-size="18" font-weight="700">{title}</text>',f'<text x="{x+22}" y="{y+90}" class="small">{sub}</text>']
        if i<3:p.append(f'<path d="M{x+w+8} 230 L{boxes[i+1][0]-12} 230" class="arrow" marker-end="url(#a)"/>')
    p += ['<text x="45" y="465" class="small">Source: models_weightedgat_temporal.py and temporal_encoders.py. Attribution needs independent validation for causal interpretation.</text>','</svg>\n']
    return "\n".join(p)


def node_svg(rows: list[dict[str, object]]) -> str:
    means={node:fmean(float(r["mean_importance"]) for r in rows if r["node"]==node) for node in NODE_ORDER}
    p=svg_start(1050,600,"Attention-derived node importance","Descriptive mean of uploaded SPAN/WeightedGAT summaries; N=1,000 coverage is partial")
    left,bottom=110,465
    for idx,node in enumerate(NODE_ORDER):
        x=150+idx*98;v=means[node];h=v/0.22*300;p += [f'<rect x="{x-25}" y="{bottom-h:.1f}" width="50" height="{h:.1f}" rx="4" fill="#00897b"/>',f'<text x="{x}" y="495" class="small" text-anchor="end" transform="rotate(-38 {x} 495)">{node}</text>']
    p += ['<line x1="110" y1="465" x2="1010" y2="465" class="axis"/>','<text x="110" y="560" class="small">Source: uploaded node-importance CSVs. Attention-derived importance is correlational and does not establish causality.</text>','</svg>\n']
    return "\n".join(p)


def expected_outputs() -> dict[Path, str]:
    compact, large = read_csv(RAW_COMPACT), read_csv(RAW_LARGE)
    validate_raw(compact, {500, 1000, 2000}); validate_raw(large, {10000, 20000})
    sample_rows, family_rows = performance_by_sample(compact), performance_by_family(large)
    aligned_large, node_rows = paper_aligned_rows(large, RAW_LARGE), node_importance_summary()
    raw_fields=list(large[0])+["evidence_type","paper_rule_scope","source_file"]
    return {
        ROOT/"results"/"final-paper-rule-set.csv": csv_text(["rule","rule_family","included_in_final_paper","evidence_type"],paper_rule_rows()),
        ROOT/"results"/"paper-reported-overall-f1.csv": csv_text(["model","overall_mean_f1","evidence_type"],paper_overall_rows()),
        ROOT/"results"/"paper-reported-large-n-scaling.csv": csv_text(["model","sample_size","auroc","f1","brier","f1_change_from_n10000","evidence_type"],paper_scaling_rows()),
        ROOT/"results"/"paper-reported-period4-f1.csv": csv_text(["model","sample_size","f1","resnet_reference_f1","paper_reported_difference_vs_resnet","arithmetic_difference_from_displayed_values","internal_consistency","evidence_type"],paper_period4_rows()),
        ROOT/"results"/"paper-aligned-large-n-raw-results.csv": csv_text(raw_fields,aligned_large),
        ROOT/"results"/"gnn-performance-by-sample-size.csv": csv_text(["sample_size","display_label","source_model","display_mapping_status","observations","mean_auroc","mean_f1_score","mean_brier_score","rule_scope","excluded_raw_rules","source_file"],sample_rows),
        ROOT/"results"/"gnn-performance-by-rule-family.csv": csv_text(["rule_family","display_label","source_model","display_mapping_status","observations","mean_auroc","mean_f1_score","mean_brier_score","sample_sizes","rule_scope","source_file"],family_rows),
        ROOT/"results"/"node-importance-by-sample-size.csv": csv_text(["sample_size","node","mean_importance","observations","rule_count","coverage","interpretation","source_file"],node_rows),
        ROOT/"results"/"result-catalog.csv": csv_text(["file","kind","rows","sample_size_count","rule_count","model_count","node_count","status"],result_catalog()),
        ROOT/"assets"/"research-evolution.svg": research_evolution_svg(),
        ROOT/"assets"/"large-n-scaling.svg": line_chart("GNN scaling from N=10,000 to N=20,000","Paper-reported mean F1 across the final 12 tasks",{m:(next(f for mm,n,a,f,b,d in PAPER_SCALING if mm==m and n==10000),next(f for mm,n,a,f,b,d in PAPER_SCALING if mm==m and n==20000)) for m in ("PC-GNN","SPAN","ILGNet")},"PAPER-REPORTED RESULT — ILGNet has the largest reported F1 change (+0.069); no uncertainty estimates were available."),
        ROOT/"assets"/"period4-f1.svg": period4_svg(),
        ROOT/"assets"/"performance-by-rule-family.svg": family_svg(family_rows),
        ROOT/"assets"/"performance-across-sample-sizes.svg": compact_svg(sample_rows),
        ROOT/"assets"/"span-gnn-architecture.svg": architecture_svg(),
        ROOT/"assets"/"node-importance.svg": node_svg(node_rows),
    }


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
    outputs=expected_outputs();mismatches=[]
    for path,content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:mismatches.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding="utf-8");print(f"wrote {path.relative_to(ROOT)}")
    if mismatches:
        print("Out-of-date generated artifacts:",file=sys.stderr)
        for path in mismatches:print(f"  - {path}",file=sys.stderr)
        return 1
    if args.check:print(f"verified {len(outputs)} deterministic artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
