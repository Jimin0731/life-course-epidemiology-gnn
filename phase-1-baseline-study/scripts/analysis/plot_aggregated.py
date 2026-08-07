from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_aggregated_performance():
    file_path = Path("output/FINAL_MASTER_LOG.csv")
    if not file_path.exists():
        print("Error: 'output/FINAL_MASTER_LOG.csv' Can't find a file.")
        return

    df = pd.read_csv(file_path)

    plot_dir = Path("output/plots_final")
    plot_dir.mkdir(exist_ok=True, parents=True)

    metrics = {
        "auc": "Average AUC Score",
        "f1": "Average F1 Score",
        "brier": "Average Brier Score (Lower is Better)",
    }

    model_order = ["LR", "XGBoost", "ResNet"]

    print("Generating aggregated plots...")

    for metric, ylabel in metrics.items():
        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=df,
            x="n_people",
            y=metric,
            hue="model_name",
            style="model_name",
            hue_order=model_order,
            style_order=model_order,
            markers=True,
            dashes=False,
            ci=None,
        )

        plt.title(
            f"Overall Model Performance vs. Data Size ({metric.upper()})", fontsize=14
        )
        plt.xlabel("Number of People (Log Scale)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        plt.xscale("log")

        ticks = sorted(df["n_people"].unique())
        plt.xticks(ticks, labels=[str(int(t)) for t in ticks])

        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend(title="Model")

        save_path = plot_dir / f"aggregated_{metric}_vs_npeople.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_aggregated_performance()
