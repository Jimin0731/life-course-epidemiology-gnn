import argparse
import glob
import os
from itertools import product

import pandas as pd

try:
    from Data_simulation.n_adaptive_config import (
        DEFAULT_TEMPORAL_ENCODER_ORDER,
        EPOCHS_CONFIG,
        get_hyperparameters,
    )
except ImportError:
    from n_adaptive_config import (
        DEFAULT_TEMPORAL_ENCODER_ORDER,
        EPOCHS_CONFIG,
        get_hyperparameters,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_results_file(explicit_path=None):
    if explicit_path:
        result_path = explicit_path
        if not os.path.isabs(result_path):
            result_path = os.path.join(BASE_DIR, result_path)
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Result file not found: {result_path}")
        return result_path

    patterns = [
        os.path.join(BASE_DIR, "final_model_comparison_metrics_V4_N*.csv"),
        os.path.join(BASE_DIR, "final_model_comparison_metrics_V*.csv"),
        os.path.join(BASE_DIR, "final_model_comparison_metrics*.csv"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError("Could not find any final_model_comparison_metrics*.csv file.")

    return max(set(candidates), key=os.path.getmtime)


def encoder_sort_key(encoder_name):
    if encoder_name in DEFAULT_TEMPORAL_ENCODER_ORDER:
        return (0, DEFAULT_TEMPORAL_ENCODER_ORDER.index(encoder_name))
    return (1, encoder_name)


def build_hyperparameter_table(results_df):
    required_columns = {"N", "Temporal_Encoder"}
    missing = required_columns - set(results_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Results file is missing required columns: {missing_text}")

    combos = results_df[["N", "Temporal_Encoder"]].dropna().drop_duplicates().copy()
    combos["_encoder_order"] = combos["Temporal_Encoder"].map(encoder_sort_key)
    combos = combos.sort_values(by=["N", "_encoder_order"]).drop(columns="_encoder_order")

    rows = []
    for combo in combos.itertuples(index=False):
        n_size = int(combo.N)
        temporal_encoder = str(combo.Temporal_Encoder)
        hparams = get_hyperparameters(n_size, temporal_encoder)
        rows.append(
            {
                "N": n_size,
                "Temporal_Encoder": temporal_encoder,
                "hidden_dim": hparams["hidden_dim"],
                "batch_size": hparams["batch_size"],
                "lr": hparams["lr"],
                "patience": hparams["patience"],
                "epochs": EPOCHS_CONFIG.get(n_size, ""),
            }
        )

    return pd.DataFrame(rows)


def build_hyperparameter_table_from_config(n_sizes, temporal_encoders=None):
    if temporal_encoders is None:
        temporal_encoders = DEFAULT_TEMPORAL_ENCODER_ORDER

    rows = []
    for n_size, temporal_encoder in product(n_sizes, temporal_encoders):
        hparams = get_hyperparameters(int(n_size), temporal_encoder)
        rows.append(
            {
                "N": int(n_size),
                "Temporal_Encoder": temporal_encoder,
                "hidden_dim": hparams["hidden_dim"],
                "batch_size": hparams["batch_size"],
                "lr": hparams["lr"],
                "patience": hparams["patience"],
                "epochs": EPOCHS_CONFIG.get(int(n_size), ""),
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print an N x temporal_encoder hyperparameter table from Run_all_n_temporal_v3 results."
    )
    parser.add_argument(
        "--results",
        help="Path to a results CSV. Defaults to the most recently modified final_model_comparison_metrics*.csv file.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the generated hyperparameter table as CSV.",
    )
    parser.add_argument(
        "--n-sizes",
        nargs="+",
        type=int,
        help="Explicit N sizes to print, independent of the contents of the results CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_sizes:
        table_df = build_hyperparameter_table_from_config(args.n_sizes)
        print(f"N sizes from config: {', '.join(map(str, args.n_sizes))}")
    else:
        results_path = resolve_results_file(args.results)
        results_df = pd.read_csv(results_path)
        table_df = build_hyperparameter_table(results_df)
        print(f"Results file: {results_path}")

    if table_df.empty:
        print("No N x Temporal_Encoder combinations found in the results file.")
        return

    print()
    print(table_df.to_string(index=False))

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(BASE_DIR, output_path)
        table_df.to_csv(output_path, index=False)
        print()
        print(f"Saved table to: {output_path}")


if __name__ == "__main__":
    main()
