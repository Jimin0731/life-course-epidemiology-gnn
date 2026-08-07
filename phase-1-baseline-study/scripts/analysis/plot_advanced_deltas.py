from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_advanced_pairwise_deltas():
    file_path = Path("output/FINAL_MASTER_LOG.csv")
    if not file_path.exists():
        print("Error: 'output/FINAL_MASTER_LOG.csv' Can't find a file.")
        return

    df = pd.read_csv(file_path)


    metrics_config = {
        "auc": {"label": "AUC Score", "direction": 1},
        "f1": {"label": "F1 Score", "direction": 1},
        "recall": {"label": "Recall (TPR)", "direction": 1},
        "fpr": {"label": "False Positive Rate", "direction": -1},  # 낮을수록 좋음
        "brier": {"label": "Brier Score", "direction": -1},  # 낮을수록 좋음
        "train_time": {"label": "Training Time (s)", "direction": -1},  # 낮을수록 좋음
    }

    comparisons = [("XGBoost", "LR"), ("ResNet", "XGBoost"), ("ResNet", "LR")]

    output_dir = Path("output/plots_final/advanced_deltas")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating advanced pairwise delta plots for ALL metrics...")

    for metric, config in metrics_config.items():
        if metric not in df.columns:
            print(f"Skipping {metric} (Column not found in CSV)")
            continue

        print(f"Processing metric: {metric}...")

        subset = df[["data_name", "n_people", "model_name", metric]]
        pivot_df = subset.pivot_table(
            index=["data_name", "n_people"], columns="model_name", values=metric
        ).reset_index()

        for model_a, model_b in comparisons:
            if model_a not in pivot_df.columns or model_b not in pivot_df.columns:
                continue

            delta_col = "delta"
            plot_data = pivot_df.copy()
            plot_data[delta_col] = plot_data[model_a] - plot_data[model_b]
            plot_data = plot_data.dropna(subset=[delta_col])

            plt.figure(figsize=(10, 6))

            sns.lineplot(
                data=plot_data,
                x="n_people",
                y=delta_col,
                marker="o",
                linewidth=2.5,
                ci=95,
                label=f"Avg. Diff ({model_a} - {model_b})",
            )

            plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

            y_min, y_max = plot_data[delta_col].min(), plot_data[delta_col].max()
            limit = max(abs(y_min), abs(y_max), 0.1) * 1.2

            if config["direction"] == 1:  # 높을수록 좋음 (AUC, F1)
                plt.fill_between(
                    [50, 2000],
                    0,
                    limit,
                    color="green",
                    alpha=0.05,
                    label=f"{model_a} Better",
                )
                plt.fill_between(
                    [50, 2000],
                    -limit,
                    0,
                    color="red",
                    alpha=0.05,
                    label=f"{model_b} Better",
                )
            else:  # 낮을수록 좋음 (Brier, FPR, Time)
                plt.fill_between(
                    [50, 2000],
                    -limit,
                    0,
                    color="green",
                    alpha=0.05,
                    label=f"{model_a} Better (Lower)",
                )
                plt.fill_between(
                    [50, 2000],
                    0,
                    limit,
                    color="red",
                    alpha=0.05,
                    label=f"{model_b} Better (Lower)",
                )

            plt.title(
                f"Performance Gap: {model_a} vs. {model_b} ({config['label']})",
                fontsize=14,
            )
            plt.xlabel("Number of People (Log Scale)", fontsize=12)
            plt.ylabel(f"Difference ({model_a} - {model_b})", fontsize=12)

            plt.xscale("log")
            ticks = sorted(df["n_people"].unique())
            plt.xticks(ticks, labels=[str(int(t)) for t in ticks])
            plt.ylim(-limit, limit)
            plt.grid(True, which="both", ls="--", alpha=0.6)
            plt.legend(loc="upper left" if config["direction"] == 1 else "lower left")

            # 저장
            file_name = f"delta_{metric}_{model_a}_vs_{model_b}.png"
            plt.savefig(output_dir / file_name, dpi=300)
            plt.close()

    print(f"\nAll plots saved in '{output_dir}'")


if __name__ == "__main__":
    plot_advanced_pairwise_deltas()
