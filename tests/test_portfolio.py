from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "06a9c193d85b655bb82a9224bbd1086777dd9755"


class PortfolioIntegrityTests(unittest.TestCase):
    def test_required_structure(self) -> None:
        required = [
            "README.md",
            "phase-1-baseline-study",
            "phase-2-gnn-extension",
            "assets",
            "results",
            "reports",
            "docs",
            "tests",
            ".github/workflows/ci.yml",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_canonical_grid_is_complete(self) -> None:
        path = ROOT / "phase-2-gnn-extension/results/raw/final_model_comparison_metrics_V4_N500_1000_2000.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(row["N"], row["Rule"], row["Model"]) for row in rows}
        self.assertEqual(len(rows), 126)
        self.assertEqual(len(keys), 126)
        self.assertEqual({int(row["N"]) for row in rows}, {500, 1000, 2000})
        self.assertEqual(len({row["Rule"] for row in rows}), 14)
        self.assertEqual(len({row["Model"] for row in rows}), 3)

    def test_committed_svgs_are_well_formed_and_labelled(self) -> None:
        expected = {
            "research-evolution.svg": "Research evolution",
            "performance-across-sample-sizes.svg": "Model performance across sample sizes",
            "performance-by-rule-family.svg": "Performance by life-course rule family",
            "span-gnn-architecture.svg": "Schematic",
            "node-importance.svg": "Attention-derived node importance",
        }
        for filename, marker in expected.items():
            path = ROOT / "assets" / filename
            ET.parse(path)
            self.assertIn(marker.lower(), path.read_text(encoding="utf-8").lower())

    def test_original_manifest_matches_source_commit(self) -> None:
        manifest = ROOT / "docs/original-file-manifest.sha256"
        entries: list[tuple[str, str]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            digest, path = line.split("  ", 1)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            entries.append((digest, path))
        self.assertEqual(len(entries), 67)
        for expected_digest, path in entries:
            content = subprocess.check_output(
                ["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT
            )
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected_digest, path)

    def test_no_machine_paths_or_likely_secrets(self) -> None:
        forbidden = [
            re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
            re.compile(rb"AKIA[0-9A-Z]{16}"),
            re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
        ]
        extensions = {".py", ".R", ".md", ".yml", ".yaml", ".txt", ".sh", ".csv", ".svg"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix not in extensions:
                continue
            content = path.read_bytes()
            for pattern in forbidden:
                self.assertIsNone(pattern.search(content), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
