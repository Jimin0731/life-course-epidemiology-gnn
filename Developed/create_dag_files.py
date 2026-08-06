import pandas as pd
import os

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

node_names = [
    "sex",          # 0
    "ethnicity",    # 1
    "marriage",     # 2
    "education",    # 3
    "class",        # 4
    "admissions",   # 5
    "admissions2",  # 6
    "admissions3",  # 7
    "admissions4"   # 8
]


# admissions2 <- class, education
# admissions3 <- sex, ethnicity
# admissions4 <- admissions2, admissions
edge_data = []

# admissions2 (index 6) depends on class (4) and education (3)
edge_data.append({"src": "class", "dst": "admissions2"})
edge_data.append({"src": "education", "dst": "admissions2"})

# admissions3 (index 7) depends on sex (0) and ethnicity (1)
edge_data.append({"src": "sex", "dst": "admissions3"})
edge_data.append({"src": "ethnicity", "dst": "admissions3"})

# admissions4 (index 8) depends on admissions2 (6) and admissions (5)
edge_data.append({"src": "admissions2", "dst": "admissions4"})
edge_data.append({"src": "admissions", "dst": "admissions4"})

node_file_path = os.path.join(data_dir, "dag_node_names.txt")
with open(node_file_path, "w") as f:
    for name in node_names:
        f.write(name + "\n")

edge_file_path = os.path.join(data_dir, "dag_edges.csv")
df_edges = pd.DataFrame(edge_data)
df_edges.to_csv(edge_file_path, index=False)

print("="*30)
print("file generation complete!")
print(f"1. {node_file_path}")
print(f"2. {edge_file_path}")
print("="*30)
print("preview generated edge data:")
print(df_edges)