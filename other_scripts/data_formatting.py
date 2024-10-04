import pandas as pd
import numpy as np
from io import StringIO

file_path = "../data/welddb/welddb.data"
print(file_path)
with open(file_path, "r") as file:
    cleaned_lines = [" ".join(line.split()) for line in file]

data = pd.read_csv(
    StringIO("\n".join(cleaned_lines)), sep=" ", na_values="N", header=None
)
print("Le nombre de lignes est: " + str(data.shape[0]))
print("Le nombre de colonnes est: " + str(data.shape[1]))

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

data.columns = columns
data.replace("N", np.nan, inplace=True)
data.loc[data["nit_c"].str.contains("tot", na=False), "nit_c"] = data["nit_c"].str[:2]
data.loc[data["inter_temp"].str.contains("150-200", na=False), "inter_temp"] = "175"
data.loc[data["hardness"].str.contains("Hv", na=False), "hardness"] = data[
    "nit_c"
].str.split("Hv", expand=True)[0]


def clean_numeric_values(value):
    if isinstance(value, str) and "<" in value:
        return float(value.replace("<", "").strip())
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


data.iloc[:, :-1] = data.iloc[:, :-1].map(clean_numeric_values)
data.set_index(["weld_id"], inplace=True)
data.to_csv("../data/welddb.csv")
