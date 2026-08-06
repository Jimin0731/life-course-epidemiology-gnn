from pathlib import Path

import pandas as pd


def parse_and_aggregate():
    output_dir = Path("output")
    all_files = list(output_dir.glob("model_output_*.csv"))

    if not all_files:
        print("Error: output 폴더에 CSV 파일이 없습니다.")
        return

    print(f"Found {len(all_files)} log files. Processing...")

    aggregated_data = []

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            row = df.iloc[0]

            # 1. 테스트 세트 샘플 수 계산
            tn = row.get("true_neg", 0)
            fp = row.get("false_pos", 0)
            fn = row.get("false_neg", 0)
            tp = row.get("true_pos", 0)

            test_set_size = tn + fp + fn + tp

            # 2. n_people 역추적 (Test set is 20% of total)
            # 데이터가 너무 작아 계산이 딱 떨어지지 않을 경우를 대비해 반올림
            n_people_estimated = int(round(test_set_size * 5, -1))

            # 50명 이하 단위(예: 50) 처리를 위해 로직 보정
            if n_people_estimated == 0:
                n_people_estimated = 50  # 혹시 모를 예외 처리

            # 3. FPR 계산
            if (fp + tn) > 0:
                fpr = fp / (fp + tn)
            else:
                fpr = 0.0

            # 4. F1 Score 추출 (우선순위: f1 > f1_nothres)
            f1 = row.get("f1")
            if pd.isna(f1):
                f1 = row.get("f1_nothres")

            # 5. 데이터 취합
            entry = {
                "data_name": row.get("data_name"),
                "model_name": row.get("model_name"),
                "n_people": n_people_estimated,  # 추정된 인원 수
                "auc": row.get("auc"),
                "f1": f1,
                "recall": row.get("recall"),  # TPR
                "fpr": fpr,
                "brier": row.get("brier"),  # <--- Brier Score  (Calibrated)
                "brier_nocal": row.get("brier_nocal"),
                "train_time": row.get("train_time"),
                "source_file": file_path.name,
            }
            aggregated_data.append(entry)

        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    # 데이터프레임 생성 및 정렬
    final_df = pd.DataFrame(aggregated_data)

    final_df = final_df.sort_values(
        by=["n_people", "data_name", "model_name"], ascending=[False, True, True]
    )

    # 저장
    save_path = output_dir / "FINAL_MASTER_LOG.csv"
    final_df.to_csv(save_path, index=False)

    print("\n==========================================")
    print(f"Success! Combined {len(final_df)} logs.")
    print(f"Saved to: {save_path}")
    print("Check 'n_people' column to ensure sizes are correct (1000, 500, etc.)")
    print("==========================================")
    print(final_df[["n_people", "data_name", "model_name", "auc", "f1"]].head())


if __name__ == "__main__":
    parse_and_aggregate()
