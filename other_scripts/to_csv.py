import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/welddb/welddb.data", sep=" ", on_bad_lines="skip")
columns = [
    "c_c",
    "si_c",
    "mn_c",
    "su_c",
    "ph_c",
    "ni_c",
    "ch_c",
    "mol_c",
    "va_c",
    "cop_c",
    "cob_c",
    "tu_c",
    "o_c",
    "ti_c",
    "nit_c",
    "al_c",
    "bo_c",
    "nio_c",
    "tin_c",
    "as_c",
    "an_c",
    "current",
    "voltage",
    "ac_dc",
    "electrode",
    "heat_in",
    "inter_temp",
    "weld_type",
    "post_weld_temp",
    "post_weld_time",
    "yield",
    "ult_tens_str",
    "elongation",
    "reduc_area",
    "charpy_temp",
    "charpy_imp_tough",
    "hardness",
    "fatt",
    "prim_ferr",
    "ferr_sec",
    "acic_ferr",
    "martensite",
    "ferr_carb",
    "weld_id",
]
cat_cols = ["ac_dc", "electrode", "weld_type", "weld_id"]
df.columns = columns
df.replace("N", np.nan, inplace=True)

df.loc[df["nit_c"].str.contains("tot", na=False), "nit_c"] = df["nit_c"].str[:2]
df.loc[df["inter_temp"].str.contains("150-200", na=False), "inter_temp"] = "175"
df.loc[df["hardness"].str.contains("Hv", na=False), "hardness"] = df["nit_c"].str.split(
    "Hv", expand=True
)[0]

# count all N for each column
for col in columns:
    if df[col].dtype == "object":
        df.loc[df[col].str.startswith("<", na=False), col] = df[col].str[1:]
    if col not in cat_cols:
        try:
            df[col] = df[col].astype(float)
        except:
            print(col)
            raise AssertionError
    # if df[col].dtype != "object":
    #     df[col] = df[col].astype(float)
    #     df[col].plot.hist()
    #     plt.show()
print(df.isna().sum())
df.set_index(["weld_id"], inplace=True)
df.to_csv("data/welddb.csv")
