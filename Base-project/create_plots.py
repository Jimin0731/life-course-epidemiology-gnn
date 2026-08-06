import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_auc_vs_npeople_plots():


    summary_file = Path("output/FINAL_ANALYSIS_BY_N_PEOPLE.csv")

    if not summary_file.exists():
        print(f"Error: Cannot find {summary_file}.")
        print("Check the file path of summary_file.")
        return

    try:
        df = pd.read_csv(summary_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    plot_dir = Path("output/plots")
    plot_dir.mkdir(exist_ok=True)

    models_to_plot = ['LR', 'XGBoost', 'ResNet']

    for data_name, group_df in df.groupby('data_name'):

        plt.figure(figsize=(10, 6))

        for model in models_to_plot:
            model_df = group_df[group_df['model_name'] == model]
            model_df = model_df.sort_values(by='n_people')

            valid_data = model_df.dropna(subset=['auc'])

            if not valid_data.empty:
                plt.plot(
                    valid_data['n_people'], 
                    valid_data['auc'], 
                    marker='o', 
                    linestyle='-', 
                    label=model
                )

        plt.title(f"AUC vs. Data Size ('{data_name}' dataset)")
        plt.xlabel("Number of People (n_people) - Log Scale")
        plt.ylabel("Average AUC Score")
        plt.legend()
        plt.grid(True, which="both", ls="--")

        plt.xscale('log')

        ticks = sorted(df['n_people'].unique())
        plt.xticks(ticks, labels=[str(t) for t in ticks])

 
        plot_filename = plot_dir / f"plot_auc_vs_npeople_{data_name}.png"
        plt.savefig(plot_filename)
        plt.close() 

    print(f"\n {len(df['data_name'].unique())} plots are saved in 'output/plots'.")
    print(f"e.g.: {plot_dir / 'plot_auc_vs_npeople_Repeats1.png'}")

if __name__ == "__main__":
    create_auc_vs_npeople_plots()