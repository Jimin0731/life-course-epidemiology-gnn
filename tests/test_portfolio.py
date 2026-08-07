from __future__ import annotations

import ast
import csv
import hashlib
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "06a9c193d85b655bb82a9224bbd1086777dd9755"
FINAL_RULES = {
    "Order1", "Order2", "Order3",
    "Repeats1", "Repeats2", "Repeats3",
    "Timing1", "Timing2", "Timing3",
    "Period1", "Period2", "Period4",
}
EXCLUDED = {"Period3", "Timing4"}


def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class PortfolioIntegrityTests(unittest.TestCase):
    def test_required_structure(self) -> None:
        for relative in (
            "README.md", "phase-1-baseline-study", "phase-2-gnn-extension",
            "assets", "results", "reports", "docs", "tests", ".github/workflows/ci.yml",
        ):
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_final_paper_rule_definition_is_exact(self) -> None:
        rule_rows = rows("results/final-paper-rule-set.csv")
        self.assertEqual(len(rule_rows), 12)
        self.assertEqual({row["rule"] for row in rule_rows}, FINAL_RULES)
        self.assertTrue(EXCLUDED.isdisjoint({row["rule"] for row in rule_rows}))
        self.assertEqual({row["included_in_final_paper"] for row in rule_rows}, {"true"})

    def test_paper_aligned_layer_filters_but_does_not_relabel_raw_rows(self) -> None:
        aligned = rows("results/paper-aligned-large-n-raw-results.csv")
        self.assertEqual(len(aligned), 72)
        self.assertEqual({row["Rule"] for row in aligned}, FINAL_RULES)
        self.assertTrue(EXCLUDED.isdisjoint({row["Rule"] for row in aligned}))
        self.assertEqual({row["N"] for row in aligned}, {"10000", "20000"})
        self.assertEqual(
            {row["Model"] for row in aligned},
            {"EconomicGNN_CNN", "WeightedGAT_LSTM", "SpatioTemporalGNN"},
        )
        self.assertEqual({row["evidence_type"] for row in aligned}, {"uploaded-raw-result"})

    def test_raw_14_rule_evidence_is_preserved(self) -> None:
        compact = rows("phase-2-gnn-extension/results/raw/final_model_comparison_metrics_V4_N500_1000_2000.csv")
        self.assertEqual(len(compact), 126)
        self.assertEqual(len({(r["N"], r["Rule"], r["Model"]) for r in compact}), 126)
        self.assertEqual(len({r["Rule"] for r in compact}), 14)
        self.assertTrue(EXCLUDED <= {r["Rule"] for r in compact})
        large = rows("phase-2-gnn-extension/results/raw/final_model_comparison_metrics_V4.csv")
        self.assertEqual(len(large), 78)
        self.assertEqual(len({r["Rule"] for r in large}), 14)
        self.assertTrue(EXCLUDED <= {r["Rule"] for r in large})

    def test_paper_reported_values_are_exact_and_labelled(self) -> None:
        overall = {row["model"]: row["overall_mean_f1"] for row in rows("results/paper-reported-overall-f1.csv")}
        self.assertEqual(overall, {"ResNet": "0.954", "PC-GNN": "0.809", "SPAN": "0.679", "ILGNet": "0.649"})
        scaling = {(r["model"], r["sample_size"]): r for r in rows("results/paper-reported-large-n-scaling.csv")}
        self.assertEqual((scaling[("ILGNet", "20000")]["f1"], scaling[("ILGNet", "20000")]["f1_change_from_n10000"]), ("0.684", "+0.069"))
        period4 = {(r["model"], r["sample_size"]): r for r in rows("results/paper-reported-period4-f1.csv")}
        self.assertEqual(period4[("ILGNet", "10000")]["paper_reported_difference_vs_resnet"], "+0.149")
        inconsistent = period4[("PC-GNN", "20000")]
        self.assertEqual(inconsistent["paper_reported_difference_vs_resnet"], "+0.016")
        self.assertEqual(inconsistent["arithmetic_difference_from_displayed_values"], "+0.110")
        self.assertEqual(inconsistent["internal_consistency"], "paper-values-conflict-retained-verbatim")
        for relative in ("results/paper-reported-overall-f1.csv", "results/paper-reported-large-n-scaling.csv", "results/paper-reported-period4-f1.csv"):
            self.assertEqual({r["evidence_type"] for r in rows(relative)}, {"paper-reported-result"})

    def test_rule_family_aggregate_has_final_scope_and_expected_pattern(self) -> None:
        family_rows = rows("results/gnn-performance-by-rule-family.csv")
        self.assertEqual({r["rule_scope"] for r in family_rows}, {"final-paper-12"})
        self.assertEqual({r["sample_sizes"] for r in family_rows}, {"10000|20000"})
        means = {
            family: sum(float(r["mean_f1_score"]) for r in family_rows if r["rule_family"] == family) / 3
            for family in ("Period", "Repeats", "Timing", "Order")
        }
        self.assertGreater(means["Period"], means["Repeats"])
        self.assertGreater(means["Repeats"], means["Timing"])
        self.assertGreater(means["Timing"], means["Order"])

    def test_display_mapping_is_unchanged_and_ambiguity_is_explicit(self) -> None:
        plot_path = ROOT / "phase-2-gnn-extension/scripts/analysis/plot_results.py"
        module = ast.parse(plot_path.read_text(encoding="utf-8"))
        mapping = None
        for node in module.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODEL_NAME_MAP" for t in node.targets):
                mapping = ast.literal_eval(node.value)
        self.assertEqual(mapping, {
            "economicgnn_cnn": "PC-GNN",
            "weightedgat_lstm": "SPAN",
            "spatiotemporalgnn": "ILGNet",
        })
        audit = (ROOT / "reports/model-identity-audit.md").read_text(encoding="utf-8")
        self.assertIn("portfolio display mapping", audit)
        self.assertIn("implementation identity is unresolved", audit.lower())
        self.assertIn("HybridGNN", audit)
        derived = rows("results/gnn-performance-by-rule-family.csv")
        ilgn = [r for r in derived if r["source_model"] == "SpatioTemporalGNN"]
        self.assertTrue(ilgn)
        self.assertTrue(all("unresolved" in r["display_mapping_status"] for r in ilgn))

    def test_runner_defines_prevalence_before_oversampling(self) -> None:
        runner = ROOT / "phase-2-gnn-extension/scripts/experiments/run_temporal_experiments.py"
        module = ast.parse(runner.read_text(encoding="utf-8"))
        function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "run_experiment")
        stores = [n.lineno for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == "pos_rate"]
        loads = [n.lineno for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == "pos_rate"]
        self.assertTrue(stores and loads)
        self.assertLess(min(stores), min(loads))
        source_lines = runner.read_text(encoding="utf-8").splitlines()
        first_load_line = source_lines[min(loads) - 1]
        self.assertIn("if pos_rate < 0.05", first_load_line)
        module_assignments = [
            n for n in module.body if isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(isinstance(t, ast.Name) and t.id == "DAG_DIR" for t in getattr(n, "targets", []))
        ]
        self.assertEqual(module_assignments, [], "DAG resolution should not block module import")

    def test_readme_does_not_make_contradicted_period4_claim(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertNotIn("all three gnns outperformed resnet at both", text)
        self.assertNotIn("all gnns outperform resnet", text)
        self.assertIn("pc-gnn did so at n=20,000 but not n=10,000", text)

    def test_generated_artifacts_are_deterministic(self) -> None:
        subprocess.run([sys.executable, "scripts/build_portfolio_assets.py", "--check"], cwd=ROOT, check=True)

    def test_committed_svgs_are_well_formed_and_evidence_labelled(self) -> None:
        expected = {
            "research-evolution.svg": "SCHEMATIC",
            "large-n-scaling.svg": "PAPER-REPORTED RESULT",
            "period4-f1.svg": "Paper-reported result",
            "performance-across-sample-sizes.svg": "DERIVED FROM UPLOADED RAW CSV",
            "performance-by-rule-family.svg": "Source:",
            "span-gnn-architecture.svg": "SCHEMATIC",
            "node-importance.svg": "Source:",
        }
        for filename, marker in expected.items():
            path = ROOT / "assets" / filename
            ET.parse(path)
            self.assertIn(marker.lower(), path.read_text(encoding="utf-8").lower())

    def test_original_manifest_matches_source_commit(self) -> None:
        entries: list[tuple[str, str]] = []
        for line in (ROOT / "docs/original-file-manifest.sha256").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            digest, path = line.split("  ", 1)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            entries.append((digest, path))
        self.assertEqual(len(entries), 67)
        for expected_digest, path in entries:
            content = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT)
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected_digest, path)

    def test_no_machine_paths_or_likely_secrets(self) -> None:
        forbidden = [re.compile(rb"/Users/[A-Za-z0-9._-]+/"), re.compile(rb"AKIA[0-9A-Z]{16}"), re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")]
        extensions = {".py", ".R", ".md", ".yml", ".yaml", ".txt", ".sh", ".csv", ".svg"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix not in extensions:
                continue
            content = path.read_bytes()
            for pattern in forbidden:
                self.assertIsNone(pattern.search(content), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
