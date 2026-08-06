from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics():
    # 1. 마스터 로그 파일 로드
    data_path = Path("output/FINAL_MASTER_LOG.csv")
    if not data_path.exists():
        print("Error: FINAL_MASTER_LOG.csv not found.")
        return

    df = pd.read_csv(data_path)
    plot_dir = Path("output/plots_final")
    plot_dir.mkdir(exist_ok=True)

    # 2. 그릴 지표 목록
    metrics = {
        "auc": "Average AUC Score",
        "f1": "Average F1 Score",
        "recall": "True Positive Rate (Recall)",
        "fpr": "False Positive Rate",
        "brier": "Brier Score (Lower is Better)",
        "train_time": "Training Time (seconds)",
    }

    models = ["LR", "XGBoost", "ResNet"]

    print("Generating plots...")

    # 3. 데이터셋별, 지표별 반복 플로팅
    for data_name in df["data_name"].unique():
        group_df = df[df["data_name"] == data_name]

        for metric, label in metrics.items():
            plt.figure(figsize=(10, 6))

            for model in models:
                model_df = group_df[group_df["model_name"] == model].sort_values(
                    "n_people"
                )

                # 유효한 데이터만 (실패한 실험은 점이 찍히지 않음)
                valid = model_df.dropna(subset=[metric])

                if not valid.empty:
                    plt.plot(valid["n_people"], valid[metric], marker="o", label=model)

            plt.title(f"{metric.upper()} vs Data Size ({data_name})")
            plt.xlabel("Number of People (Log Scale)")
            plt.ylabel(label)
            plt.xscale("log")
            plt.grid(True, which="both", ls="--")
            plt.legend()

            # X축 눈금 설정
            ticks = sorted(df["n_people"].unique())
            plt.xticks(ticks, labels=[str(int(t)) for t in ticks])

            # 저장
            save_name = plot_dir / f"{data_name}_{metric}.png"
            plt.savefig(save_name)
            plt.close()

    print(f"All plots saved to {plot_dir}/")


if __name__ == "__main__":
    plot_metrics()
