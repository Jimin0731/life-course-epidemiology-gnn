from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_plots_for_metric(df, metric_to_plot, plot_dir):
    models_to_plot = ["LR", "XGBoost", "ResNet"]

    for data_name, group_df in df.groupby("data_name"):
        plt.figure(figsize=(10, 6))

        for model in models_to_plot:
            model_df = group_df[group_df["model_name"] == model]
            model_df = model_df.sort_values(by="n_people")
            valid_data = model_df.dropna(subset=[metric_to_plot])

            if not valid_data.empty:
                plt.plot(
                    valid_data["n_people"],
                    valid_data[metric_to_plot],
                    marker="o",
                    linestyle="-",
                    label=model,
                )

        plt.title(f"{metric_to_plot.upper()} vs. Data Size ('{data_name}' dataset)")
        plt.xlabel("Number of People (n_people) - Log Scale")
        plt.ylabel(f"Average {metric_to_plot.upper()} Score")
        plt.legend()
        plt.grid(True, which="both", ls="--")

        plt.xscale("log")
        ticks = sorted(df["n_people"].unique())
        plt.xticks(ticks, labels=[str(t) for t in ticks])

        plot_filename = plot_dir / f"plot_{metric_to_plot}_vs_npeople_{data_name}.png"
        plt.savefig(plot_filename)
        plt.close()


def run_all_plots():
    summary_file = Path("output/FINAL_ANALYSIS_BY_N_PEOPLE.csv")

    if not summary_file.exists():
        print(f"Error: Could not find {summary_file}. Please verify the filename.")
        return

    df = pd.read_csv(summary_file)

    plot_dir = Path("output/plots")
    plot_dir.mkdir(exist_ok=True)

    metrics_to_run = ["auc", "f1", "tpr_recall", "fpr"]

    print(
        f"Starting {len(metrics_to_run)} metric plots across {len(df['data_name'].unique())} datasets..."
    )

    for metric in metrics_to_run:
        if metric not in df.columns:
            print(f"Warning: Column '{metric}' is missing in the CSV. Skipping.")
            continue

        create_plots_for_metric(df, metric, plot_dir)
        print(f"  -> '{metric.upper()}' plot completed.")

    print("\nAll metric plots are complete!")


if __name__ == "__main__":
    run_all_plots()
