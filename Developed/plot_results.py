import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = [
    "final_model_comparison_metrics_V4_N500_1000_2000.csv",
    "final_model_comparison_metrics_V4.csv",
]

MODEL_NAME_MAP = {
    "economicgnn_cnn": "PC-GNN",
    "weightedgat_lstm": "SPAN",
    "spatiotemporalgnn": "ILGNet",
}
EXCLUDED_RULES = {"period3", "timing4"}
SCATTER_LEGEND_RULE_EXCLUSIONS = {"timing3", "period3"}


def resolve_input_file(input_name: str | None) -> Path:
    if input_name:
        input_path = Path(input_name)
        if not input_path.is_absolute():
            input_path = BASE_DIR / input_path
        if input_path.exists():
            return input_path
        raise FileNotFoundError(f"No results file found: {input_path}")

    for candidate in DEFAULT_CANDIDATES:
        candidate_path = BASE_DIR / candidate
        if candidate_path.exists():
            return candidate_path

    matches = sorted(BASE_DIR.glob("final_model_comparison_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No results file found in {BASE_DIR}")


def resolve_output_dir(output_dir: str | None, input_path: Path) -> Path:
    if output_dir:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
        return output_path
    return BASE_DIR / "plots" / input_path.stem


def filter_scatter_legend(ax, n_val: int) -> None:
    legend = ax.get_legend()
    if legend is None:
        return

    excluded_labels = set(SCATTER_LEGEND_RULE_EXCLUSIONS)
    if n_val == 500:
        excluded_labels.discard("timing3")

    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label.strip().lower() not in excluded_labels
    ]
    if not filtered:
        legend.remove()
        return

    filtered_handles, filtered_labels = zip(*filtered)
    legend.remove()
    ax.legend(filtered_handles, filtered_labels, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GNN experiment results.")
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file name or path. Defaults to the newest known results file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots. Defaults to Data_simulation/plots/<csv_stem>/",
    )
    args = parser.parse_args()

    input_path = resolve_input_file(args.input)
    output_dir = resolve_output_dir(args.output_dir, input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df_display = df.copy()
    df_display = df_display[~df_display["Rule"].astype(str).str.lower().isin(EXCLUDED_RULES)].copy()
    df_display["Model_Display"] = (
        df_display["Model"]
        .astype(str)
        .str.lower()
        .map(MODEL_NAME_MAP)
        .fillna(df_display["Model"])
    )

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_display,
        x="N",
        y="AUROC",
        hue="Model_Display",
        style="Model_Display",
        markers=True,
        dashes=False,
        linewidth=2.5,
    )
    plt.title("Impact of Sample Size (N) on Model Performance", fontsize=15)
    plt.ylabel("Average AUROC Score", fontsize=12)
    plt.xlabel("Sample Size (N)", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plot1_path = output_dir / "1_performance_by_N.png"
    plt.savefig(plot1_path, dpi=300)
    print(f"Graph saved: {plot1_path}")
    plt.close()

    n_values = sorted(df_display["N"].unique())

    for n_val in n_values:
        df_n = df_display[df_display["N"] == n_val]

        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_n, x="Rule", y="AUROC", hue="Model_Display", palette="viridis")
        plt.title(f"Model Comparison by Rule (at N={n_val})", fontsize=15)
        plt.ylabel("AUROC Score", fontsize=12)
        plt.xlabel("LCE Rules", fontsize=12)
        plt.ylim(0.4, 1.0)
        plt.axhline(0.5, color="red", linestyle="--", label="Random Guess")
        plt.xticks(rotation=45)
        plt.legend(title="Model", loc="lower right")
        plt.tight_layout()
        plot2_path = output_dir / f"2_model_comparison_by_rule_N{n_val}.png"
        plt.savefig(plot2_path, dpi=300)
        print(f"Graph saved: {plot2_path}")
        plt.close()

        plt.figure(figsize=(8, 8))
        ax = sns.scatterplot(data=df_n, x="AUROC", y="F1_Score", hue="Model_Display", style="Rule", s=100)
        filter_scatter_legend(ax, n_val)
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title(f"AUROC vs F1 Score (N={n_val})", fontsize=15)
        plt.xlim(0.4, 1.0)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plot3_path = output_dir / f"3_auroc_vs_f1_N{n_val}.png"
        plt.savefig(plot3_path, dpi=300)
        print(f"Graph saved: {plot3_path}")
        plt.close()


if __name__ == "__main__":
    main()
