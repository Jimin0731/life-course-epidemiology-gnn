import pandas as pd

df = pd.read_csv("/Users/jiminbyun/Risk_Factor_simulation_DL/output/FINAL_MASTER_LOG.csv")
result = df[(df['n_people']==500) & (df['data_name']=='Order1')]
print(result)