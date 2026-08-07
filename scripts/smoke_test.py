#!/usr/bin/env python3
"""Run the deterministic, dependency-light repository smoke path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "scripts/build_portfolio_assets.py", "--check")
    run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
