"""
Merge model output CSV files into one CSV.

Usage:
    python src/lce/merge_model_results.py
    python src/lce/merge_model_results.py --input-dir output/model_results --output-file merged.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_input_dir(path: Path) -> Path:
    """Resolve input dir against cwd first, then project root."""
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    root_candidate = PROJECT_ROOT / path
    return root_candidate.resolve()


def merge_result_csvs(input_dir: Path, output_file: str = "model_output_merged.csv") -> Path:
    input_dir = resolve_input_dir(input_dir)
    csv_files = sorted(input_dir.glob("model_output_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No files matched: {input_dir}/model_output_*.csv "
            f"(cwd={Path.cwd()})"
        )

    rows = []
    file_pattern = re.compile(
        r"^model_output_(?P<data_name>.+)_n(?P<n_people>\d+)_(?P<model_name>[^_]+)"
    )

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        row = df.iloc[[0]].copy()
        row["source_file"] = file_path.name

        match = file_pattern.match(file_path.stem)
        if match:
            if "data_name" not in row.columns or pd.isna(row.iloc[0].get("data_name")):
                row["data_name"] = match.group("data_name")
            if "model_name" not in row.columns or pd.isna(row.iloc[0].get("model_name")):
                row["model_name"] = match.group("model_name")
            row["n_people"] = int(match.group("n_people"))

        rows.append(row)

    if not rows:
        raise ValueError("CSV files exist but no usable rows were found.")

    merged_df = pd.concat(rows, ignore_index=True, sort=False)
    sort_cols = [col for col in ["n_people", "data_name", "model_name"] if col in merged_df]
    if sort_cols:
        merged_df = merged_df.sort_values(by=sort_cols, ascending=[False, True, True])

    output_path = input_dir / output_file
    merged_df.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge model_output_*.csv files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "output/model_results",
        help="Directory containing model_output_*.csv files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="model_output_merged.csv",
        help="Output CSV file name",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merged_path = merge_result_csvs(args.input_dir, args.output_file)
    print(f"Merged CSV saved to: {merged_path}")


if __name__ == "__main__":
    main()
