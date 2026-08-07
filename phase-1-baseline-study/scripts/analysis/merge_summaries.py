import pandas as pd
from pathlib import Path
import re

def merge_all_summaries():
    output_dir = Path("output")
    summary_files = list(output_dir.glob("summary_*.csv")) 

    if not summary_files:
        print("Error: Cannot find 'output/summary_*.csv'.")
        return

    all_data = []

    for file_path in summary_files:
        match = re.search(r'summary_(\d+)\.csv', file_path.name)
        if not match:
            print(f"Warning: Cannot find n_people values in {file_path.name}, skipping.")
            continue

        n_people = int(match.group(1))

        try:
            df = pd.read_csv(file_path)
            df['n_people'] = n_people 
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_data:
        print("No data to merge.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    final_df = final_df.sort_values(by=['n_people', 'data_name', 'model_name'], ascending=[False, True, True])

    final_path = output_dir / "FINAL_ANALYSIS_BY_N_PEOPLE.csv"
    final_df.to_csv(final_path, index=False)

    print(f"\nMerge completed! Svaed at {final_path}.")
    print(final_df.head())

if __name__ == "__main__":
    merge_all_summaries()