import pandas as pd

df = pd.read_csv("data/welddb/welddb.data", sep=" ", on_bad_lines="skip")
print(df.shape)
columns = list(range(df.shape[1]))
print(columns)
df.columns = columns
df.to_csv("data/welddb.csv")
