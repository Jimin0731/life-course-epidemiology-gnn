import pandas as pd
from pathlib import Path

def parse_all_logs():

    output_dir = Path("output")
    log_files = list(output_dir.glob("model_output_*.csv"))

    if not log_files:
        print(f"Error: Can not find 'model_output_*.csv' files in 'output/' folder.")
        return

    print(f"{len(log_files)} parsing...")

    all_results = []

    for file_path in log_files:
        try:
            df = pd.read_csv(file_path)

            row = df.iloc[0]

            false_pos = row.get('false_pos', 0)
            true_neg = row.get('true_neg', 0)

            if (false_pos + true_neg) > 0:
                fpr = false_pos / (false_pos + true_neg)
            else:
                fpr = 0.0

            summary = {
                'data_name': row.get('data_name'),
                'model_name': row.get('model_name'),
                'auc': row.get('auc'),
                'tpr_recall': row.get('recall'), 
                'fpr': fpr,
                'stoc': row.get('stoc'),
                'randnum_train': row.get('randnum_train'),
                'train_time': row.get('train_time'),
                'source_file': file_path.name
            }
            all_results.append(summary)

        except Exception as e:
            print(f"Warning: error occurs with parsing {file_path.name} : {e}")

    if not all_results:
        print("No data for parsing.")
        return

    final_df = pd.DataFrame(all_results)

    final_summary_path = output_dir / "summary_all_runs.csv"
    final_df.to_csv(final_summary_path, index=False)

    print("\n========================================")
    print(f"Parsing completed! Saved at {final_summary_path} .")
    print("========================================")
    print(final_df.head())


if __name__ == "__main__":
    parse_all_logs()