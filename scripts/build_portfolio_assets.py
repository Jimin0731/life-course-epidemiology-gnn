#!/usr/bin/env python3
"""Build deterministic CSV summaries and dependency-free SVG portfolio figures."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "phase-2-gnn-extension" / "results" / "raw"
CANONICAL = RAW / "final_model_comparison_metrics_V4_N500_1000_2000.csv"
MODEL_ORDER = ["EconomicGNN_CNN", "WeightedGAT_LSTM", "SpatioTemporalGNN"]
MODEL_NAMES = {
    "EconomicGNN_CNN": "PC-GNN",
    "WeightedGAT_LSTM": "SPAN",
    "SpatioTemporalGNN": "ILGNet",
}
COLORS = {
    "EconomicGNN_CNN": "#1967d2",
    "WeightedGAT_LSTM": "#00897b",
    "SpatioTemporalGNN": "#e76f51",
}
NODE_ORDER = [
    "sex",
    "ethnicity",
    "marriage",
    "education",
    "class",
    "admissions",
    "admissions2",
    "admissions3",
    "admissions4",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def validate_canonical(rows: list[dict[str, str]]) -> None:
    required = {
        "N",
        "Rule",
        "Model",
        "Temporal_Encoder",
        "AUROC",
        "F1_Score",
        "Brier_Score",
    }
    if not rows or required - set(rows[0]):
        raise ValueError("Canonical result file is empty or missing required columns")
    keys = {(row["N"], row["Rule"], row["Model"]) for row in rows}
    if len(rows) != 126 or len(keys) != 126:
        raise ValueError("Expected 126 unique N × Rule × Model rows")
    if {int(row["N"]) for row in rows} != {500, 1000, 2000}:
        raise ValueError("Unexpected sample-size grid")
    if {row["Model"] for row in rows} != set(MODEL_ORDER):
        raise ValueError("Unexpected model grid")
    if len({row["Rule"] for row in rows}) != 14:
        raise ValueError("Expected 14 life-course rules")


def performance_summaries(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_n: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    by_family: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row["Model"]
        n_value = int(row["N"])
        family_match = re.match(r"[A-Za-z]+", row["Rule"])
        if family_match is None:
            raise ValueError(f"Cannot derive rule family from {row['Rule']!r}")
        by_n[(n_value, model)].append(row)
        by_family[(family_match.group(0), model)].append(row)

    def aggregate(group: list[dict[str, str]]) -> dict[str, object]:
        return {
            "observations": len(group),
            "mean_auroc": f"{fmean(float(row['AUROC']) for row in group):.6f}",
            "mean_f1_score": f"{fmean(float(row['F1_Score']) for row in group):.6f}",
            "mean_brier_score": f"{fmean(float(row['Brier_Score']) for row in group):.6f}",
        }

    sample_rows: list[dict[str, object]] = []
    for n_value in (500, 1000, 2000):
        for model in MODEL_ORDER:
            sample_rows.append(
                {
                    "sample_size": n_value,
                    "portfolio_model": MODEL_NAMES[model],
                    "source_model": model,
                    **aggregate(by_n[(n_value, model)]),
                    "source_file": CANONICAL.name,
                }
            )

    family_order = ["Order", "Period", "Repeats", "Timing"]
    family_rows: list[dict[str, object]] = []
    for family in family_order:
        for model in MODEL_ORDER:
            family_rows.append(
                {
                    "rule_family": family,
                    "portfolio_model": MODEL_NAMES[model],
                    "source_model": model,
                    **aggregate(by_family[(family, model)]),
                    "sample_sizes": "500|1000|2000",
                    "source_file": CANONICAL.name,
                }
            )
    return sample_rows, family_rows


def node_importance_summary() -> list[dict[str, object]]:
    source_files = {
        500: RAW / "node_importance_summary_N500.csv",
        1000: RAW / "node_importance_summary_N1000.csv",
        2000: RAW / "node_importance_summary_N2000.csv",
    }
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for n_value, source in source_files.items():
        for row in read_csv(source):
            grouped[(n_value, row["Node"])].append(row)

    output: list[dict[str, object]] = []
    for n_value in (500, 1000, 2000):
        for node in NODE_ORDER:
            group = grouped[(n_value, node)]
            output.append(
                {
                    "sample_size": n_value,
                    "node": node,
                    "mean_importance": f"{fmean(float(row['Importance']) for row in group):.6f}",
                    "observations": len(group),
                    "rule_count": len({row["Rule"] for row in group}),
                    "coverage": "partial" if n_value == 1000 else "available-set-complete",
                    "source_file": source_files[n_value].name,
                }
            )
    return output


def result_catalog() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in sorted(RAW.iterdir(), key=lambda item: item.name.lower()):
        if path.name == "README.md":
            continue
        if path.suffix.lower() != ".csv":
            output.append(
                {
                    "file": path.name,
                    "kind": "log",
                    "rows": 0,
                    "sample_size_count": 0,
                    "rule_count": 0,
                    "model_count": 0,
                    "node_count": 0,
                    "status": "empty" if path.stat().st_size == 0 else "present",
                }
            )
            continue
        rows = read_csv(path)
        columns = set(rows[0]) if rows else set()
        kind = "node_importance" if "Importance" in columns else "model_performance"
        output.append(
            {
                "file": path.name,
                "kind": kind,
                "rows": len(rows),
                "sample_size_count": len({row.get("N", "") for row in rows if row.get("N", "")}),
                "rule_count": len({row.get("Rule", "") for row in rows if row.get("Rule", "")}),
                "model_count": len({row.get("Model", "") for row in rows if row.get("Model", "")}),
                "node_count": len({row.get("Node", "") for row in rows if row.get("Node", "")}),
                "status": "canonical" if path == CANONICAL else "retained-revision",
            }
        )
    return output


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{title}</title>",
        f"<desc id=\"desc\">{subtitle}</desc>",
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#183153}.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#526579}.label{font-size:14px}.small{font-size:12px;fill:#526579}.value{font-size:11px;font-weight:600}.box{stroke:#c9d5e3;stroke-width:1.5}.axis{stroke:#9eafc1;stroke-width:1}.grid{stroke:#e5ebf2;stroke-width:1}.arrow{stroke:#49647e;stroke-width:2;fill:none}</style>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        f'<text x="50" y="52" class="title">{title}</text>',
        f'<text x="50" y="78" class="sub">{subtitle}</text>',
    ]


def research_evolution_svg() -> str:
    parts = svg_start(
        1200,
        430,
        "Research evolution",
        "From a collaborative baseline benchmark to Jimin Byun's individual GNN extension",
    )
    parts += [
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#49647e"/></marker></defs>',
        '<rect x="55" y="115" width="470" height="235" rx="18" fill="#e8f1ff" class="box"/>',
        '<text x="85" y="155" font-size="15" font-weight="700" fill="#1967d2">PHASE 1 · COLLABORATIVE</text>',
        '<text x="85" y="190" font-size="23" font-weight="700">Baseline benchmark</text>',
        '<text x="85" y="225" class="label">Logistic Regression · XGBoost · ResNet</text>',
        '<text x="85" y="255" class="label">Sample-size and life-course rule comparison</text>',
        '<text x="85" y="300" class="small">Alec Kemp · Jimin Byun · Yuxin Du</text>',
        '<text x="85" y="324" class="small">Jimin led implementation, execution, results, and analysis.</text>',
        '<path d="M540 232 L650 232" class="arrow" marker-end="url(#arrow)"/>',
        '<text x="557" y="214" class="small">research</text><text x="557" y="254" class="small">extension</text>',
        '<rect x="670" y="115" width="475" height="235" rx="18" fill="#e6f7f3" class="box"/>',
        '<text x="700" y="155" font-size="15" font-weight="700" fill="#00897b">PHASE 2 · INDIVIDUAL</text>',
        '<text x="700" y="190" font-size="23" font-weight="700">Graph neural networks</text>',
        '<text x="700" y="225" class="label">PC-GNN · SPAN · ILGNet</text>',
        '<text x="700" y="255" class="label">DAG structure · temporal encoding · attention</text>',
        "<text x=\"700\" y=\"300\" class=\"small\">Jimin Byun's individual extension</text>",
        '<text x="700" y="324" class="small">Raw implementation names retained for result provenance.</text>',
        '<text x="55" y="397" class="small">Schematic overview — model performance is shown only in the separate evidence-backed figures.</text>',
        "</svg>\n",
    ]
    return "\n".join(parts)


def performance_svg(rows: list[dict[str, object]]) -> str:
    width, height = 1000, 620
    left, right, top, bottom = 110, 930, 125, 500
    x_positions = {500: 190, 1000: 510, 2000: 830}
    y_min, y_max = 0.50, 0.95

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = svg_start(
        width,
        height,
        "Model performance across sample sizes",
        "Mean AUROC across all 14 life-course rules; one reported value per rule and model",
    )
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="70" y="{yy + 5:.1f}" class="small">{tick:.1f}</text>')
    parts += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        '<text x="20" y="320" class="label" transform="rotate(-90 20 320)">Mean AUROC</text>',
        '<text x="450" y="555" class="label">Sample size (N)</text>',
    ]
    indexed = {(int(row["sample_size"]), str(row["source_model"])): float(row["mean_auroc"]) for row in rows}
    for n_value, x_value in x_positions.items():
        parts.append(f'<text x="{x_value - 18}" y="525" class="label">{n_value:,}</text>')
    for model in MODEL_ORDER:
        points = [(x_positions[n], y(indexed[(n, model)]), indexed[(n, model)]) for n in x_positions]
        point_text = " ".join(f"{xv},{yv:.1f}" for xv, yv, _ in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{COLORS[model]}" stroke-width="3"/>')
        for xv, yv, value in points:
            parts.append(f'<circle cx="{xv}" cy="{yv:.1f}" r="6" fill="{COLORS[model]}"/>')
            parts.append(f'<text x="{xv + 9}" y="{yv - 9:.1f}" class="value">{value:.3f}</text>')
    legend_x = 610
    for index, model in enumerate(MODEL_ORDER):
        yy = 105 + index * 24
        parts.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 30}" y2="{yy}" stroke="{COLORS[model]}" stroke-width="4"/>')
        parts.append(f'<text x="{legend_x + 40}" y="{yy + 5}" class="small">{MODEL_NAMES[model]} ({model})</text>')
    parts += [
        '<text x="110" y="590" class="small">Source: complete V4 N=500/1,000/2,000 table (126 rows). Descriptive means; no uncertainty estimates available.</text>',
        "</svg>\n",
    ]
    return "\n".join(parts)


def rule_family_svg(rows: list[dict[str, object]]) -> str:
    width, height = 1100, 650
    left, right, top, bottom = 105, 1030, 130, 520
    y_min, y_max = 0.50, 1.00

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = svg_start(
        width,
        height,
        "Performance by life-course rule family",
        "Mean AUROC across N=500, 1,000, and 2,000, grouped by rule-name prefix",
    )
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="65" y="{yy + 5:.1f}" class="small">{tick:.1f}</text>')
    indexed = {(str(row["rule_family"]), str(row["source_model"])): float(row["mean_auroc"]) for row in rows}
    families = ["Order", "Period", "Repeats", "Timing"]
    centers = [210, 440, 670, 900]
    offsets = [-50, 0, 50]
    for family, center in zip(families, centers):
        parts.append(f'<text x="{center - 28}" y="552" class="label">{family}</text>')
        for model, offset in zip(MODEL_ORDER, offsets):
            value = indexed[(family, model)]
            yy = y(value)
            bar_height = bottom - yy
            parts.append(f'<rect x="{center + offset - 18}" y="{yy:.1f}" width="36" height="{bar_height:.1f}" rx="4" fill="{COLORS[model]}"/>')
            parts.append(f'<text x="{center + offset - 17}" y="{yy - 7:.1f}" class="value">{value:.3f}</text>')
    parts += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        '<text x="20" y="350" class="label" transform="rotate(-90 20 350)">Mean AUROC</text>',
    ]
    for index, model in enumerate(MODEL_ORDER):
        x_value = 240 + index * 280
        parts.append(f'<rect x="{x_value}" y="585" width="16" height="16" rx="3" fill="{COLORS[model]}"/>')
        parts.append(f'<text x="{x_value + 24}" y="598" class="small">{MODEL_NAMES[model]} ({model})</text>')
    parts += [
        '<text x="105" y="630" class="small">Families contain 3 Order, 4 Period, 3 Repeats, and 4 Timing rules at each sample size.</text>',
        "</svg>\n",
    ]
    return "\n".join(parts)


def architecture_svg() -> str:
    parts = svg_start(
        1200,
        560,
        "SPAN architecture",
        "Schematic of the uploaded WeightedGAT_LSTM implementation; not a measured result",
    )
    parts += [
        '<defs><marker id="arch-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#49647e"/></marker></defs>',
        '<rect x="55" y="150" width="160" height="120" rx="15" fill="#eef3f8" class="box"/>',
        '<text x="85" y="190" font-size="17" font-weight="700">Longitudinal nodes</text>',
        '<text x="83" y="220" class="small">9 variables × 40 times</text>',
        '<text x="83" y="245" class="small">DAG edge weights</text>',
        '<path d="M220 210 L285 210" class="arrow" marker-end="url(#arch-arrow)"/>',
        '<rect x="300" y="135" width="180" height="150" rx="15" fill="#e8f1ff" class="box"/>',
        '<text x="350" y="180" font-size="18" font-weight="700">LSTM encoder</text>',
        '<text x="330" y="215" class="small">Per-node temporal history</text>',
        '<text x="333" y="240" class="small">Bidirectional representation</text>',
        '<path d="M485 210 L550 210" class="arrow" marker-end="url(#arch-arrow)"/>',
        '<rect x="565" y="115" width="205" height="190" rx="15" fill="#e6f7f3" class="box"/>',
        '<text x="615" y="158" font-size="18" font-weight="700">Weighted GATv2</text>',
        '<text x="602" y="195" class="small">Edge attributes enter attention</text>',
        '<text x="606" y="225" class="small">Multi-head message passing</text>',
        '<text x="607" y="255" class="small">Attention weights retained</text>',
        '<path d="M775 210 L840 210" class="arrow" marker-end="url(#arch-arrow)"/>',
        '<rect x="855" y="135" width="150" height="150" rx="15" fill="#fff0ea" class="box"/>',
        '<text x="885" y="178" font-size="18" font-weight="700">Prediction head</text>',
        '<text x="888" y="213" class="small">Graph pooling</text>',
        '<text x="888" y="240" class="small">Binary outcome</text>',
        '<path d="M670 310 C670 370 870 355 870 400" class="arrow" marker-end="url(#arch-arrow)"/>',
        '<rect x="790" y="400" width="300" height="80" rx="15" fill="#f7f0ff" class="box"/>',
        '<text x="830" y="435" font-size="17" font-weight="700">Attention-derived importance</text>',
        '<text x="830" y="460" class="small">Descriptive interpretation, not causality</text>',
        '<text x="55" y="525" class="small">SCHEMATIC — based on temporal_encoders.py, models_weightedgat_temporal.py, and analyze_node_importance.py.</text>',
        "</svg>\n",
    ]
    return "\n".join(parts)


def node_importance_svg(rows: list[dict[str, object]]) -> str:
    width, height = 1100, 690
    left, right, top, bottom = 130, 1030, 140, 550
    n_colors = {500: "#1967d2", 1000: "#e76f51", 2000: "#00897b"}
    max_value = 0.25

    def y(value: float) -> float:
        return bottom - value / max_value * (bottom - top)

    x_positions = {node: 165 + index * 105 for index, node in enumerate(NODE_ORDER)}
    indexed = {(int(row["sample_size"]), str(row["node"])): float(row["mean_importance"]) for row in rows}
    parts = svg_start(
        width,
        height,
        "Attention-derived node importance",
        "Mean uploaded WeightedGAT importance by sample size; N=1,000 covers two rules only",
    )
    for tick in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="80" y="{yy + 5:.1f}" class="small">{tick:.2f}</text>')
    for n_value in (500, 1000, 2000):
        points = [(x_positions[node], y(indexed[(n_value, node)])) for node in NODE_ORDER]
        point_text = " ".join(f"{xv},{yv:.1f}" for xv, yv in points)
        dash = ' stroke-dasharray="7 5"' if n_value == 1000 else ""
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{n_colors[n_value]}" stroke-width="3"{dash}/>')
        for xv, yv in points:
            parts.append(f'<circle cx="{xv}" cy="{yv:.1f}" r="5" fill="{n_colors[n_value]}"/>')
    for node, x_value in x_positions.items():
        parts.append(f'<text x="{x_value - 4}" y="575" class="small" text-anchor="end" transform="rotate(-38 {x_value - 4} 575)">{node}</text>')
    parts += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        '<text x="25" y="385" class="label" transform="rotate(-90 25 385)">Mean importance</text>',
    ]
    for index, n_value in enumerate((500, 1000, 2000)):
        x_value = 240 + index * 250
        dash = ' stroke-dasharray="7 5"' if n_value == 1000 else ""
        label = f"N={n_value:,}" + (" (partial: 2 rules)" if n_value == 1000 else " (7 rules)")
        parts.append(f'<line x1="{x_value}" y1="635" x2="{x_value + 32}" y2="635" stroke="{n_colors[n_value]}" stroke-width="4"{dash}/>')
        parts.append(f'<text x="{x_value + 42}" y="640" class="small">{label}</text>')
    parts += [
        '<text x="130" y="675" class="small">Attention-derived importance describes model weighting and must not be interpreted as a causal effect.</text>',
        "</svg>\n",
    ]
    return "\n".join(parts)


def expected_outputs() -> dict[Path, str]:
    canonical_rows = read_csv(CANONICAL)
    validate_canonical(canonical_rows)
    sample_rows, family_rows = performance_summaries(canonical_rows)
    node_rows = node_importance_summary()
    catalog_rows = result_catalog()
    return {
        ROOT / "results" / "gnn-performance-by-sample-size.csv": csv_text(
            ["sample_size", "portfolio_model", "source_model", "observations", "mean_auroc", "mean_f1_score", "mean_brier_score", "source_file"],
            sample_rows,
        ),
        ROOT / "results" / "gnn-performance-by-rule-family.csv": csv_text(
            ["rule_family", "portfolio_model", "source_model", "observations", "mean_auroc", "mean_f1_score", "mean_brier_score", "sample_sizes", "source_file"],
            family_rows,
        ),
        ROOT / "results" / "node-importance-by-sample-size.csv": csv_text(
            ["sample_size", "node", "mean_importance", "observations", "rule_count", "coverage", "source_file"],
            node_rows,
        ),
        ROOT / "results" / "result-catalog.csv": csv_text(
            ["file", "kind", "rows", "sample_size_count", "rule_count", "model_count", "node_count", "status"],
            catalog_rows,
        ),
        ROOT / "assets" / "research-evolution.svg": research_evolution_svg(),
        ROOT / "assets" / "performance-across-sample-sizes.svg": performance_svg(sample_rows),
        ROOT / "assets" / "performance-by-rule-family.svg": rule_family_svg(family_rows),
        ROOT / "assets" / "span-gnn-architecture.svg": architecture_svg(),
        ROOT / "assets" / "node-importance.svg": node_importance_svg(node_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if committed artifacts differ from deterministic regeneration")
    args = parser.parse_args()
    outputs = expected_outputs()
    mismatches: list[str] = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                mismatches.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if mismatches:
        print("Out-of-date generated artifacts:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  - {mismatch}", file=sys.stderr)
        return 1
    if args.check:
        print(f"verified {len(outputs)} deterministic artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
