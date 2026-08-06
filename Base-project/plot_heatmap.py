from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_failure_heatmap():
    df = pd.read_csv("output/FINAL_MASTER_LOG.csv")

    models = ["LR", "ResNet", "XGBoost"]

    for model in models:
        model_df = df[df["model_name"] == model]

        pivot_table = model_df.pivot_table(
            index="data_name", columns="n_people", values="auc", aggfunc="mean"
        )

        pivot_table = pivot_table.sort_index(axis=1, ascending=False)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            cbar_kws={"label": "AUC Score"},
            vmin=0.4,
            vmax=1.0,
        )

        plt.title(f"Performance Heatmap: {model} (AUC)")
        plt.xlabel("Number of People (n_people)")
        plt.ylabel("Dataset Name")

        Path("output/plots_final").mkdir(exist_ok=True)
        plt.savefig(f"output/plots_final/heatmap_{model}.png")
        print(f"Saved heatmap for {model}")


if __name__ == "__main__":
    plot_failure_heatmap()
