from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pairwise_deltas():
    file_path = Path("output/FINAL_MASTER_LOG.csv")
    if not file_path.exists():
        print("Error: 'output/FINAL_MASTER_LOG.csv' Can't find a file.")
        return

    df = pd.read_csv(file_path)

    comparisons = [("XGBoost", "LR"), ("ResNet", "XGBoost"), ("ResNet", "LR")]

    subset = df[["data_name", "n_people", "model_name", "auc"]]
    pivot_df = subset.pivot_table(
        index=["data_name", "n_people"], columns="model_name", values="auc"
    ).reset_index()

    output_dir = Path("output/plots_final/deltas")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating pairwise delta plots...")

    for model_a, model_b in comparisons:
        if model_a not in pivot_df.columns or model_b not in pivot_df.columns:
            print(f"Skipping {model_a} vs {model_b} (Missing data)")
            continue

        delta_col = f"{model_a}_minus_{model_b}"
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
            label=f"Average Improvement ({model_a} - {model_b})",
        )

        plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        plt.fill_between(
            [50, 2000], 0, 1.0, color="green", alpha=0.05, label=f"{model_a} is Better"
        )
        plt.fill_between(
            [50, 2000], -1.0, 0, color="red", alpha=0.05, label=f"{model_b} is Better"
        )

        plt.title(f"Performance Gap: {model_a} vs. {model_b} (AUC)", fontsize=14)
        plt.xlabel("Number of People (Log Scale)", fontsize=12)
        plt.ylabel(f"AUC Difference ({model_a} - {model_b})", fontsize=12)

        plt.xscale("log")
        ticks = sorted(df["n_people"].unique())
        plt.xticks(ticks, labels=[str(int(t)) for t in ticks])
        plt.ylim(-0.5, 0.5)
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend(loc="upper left")

        save_path = output_dir / f"plot_delta_{model_a}_vs_{model_b}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    print("\nAll pairwise plots saved in 'output/plots_final/deltas/'")


if __name__ == "__main__":
    plot_pairwise_deltas()
