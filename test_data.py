import pandas as pd
import json
import re

# liste des colonnes
colonnes = ["Carbon concentration", 
            "Silicon concentration",
            "Manganese concentration",
            "Sulphur concentration",
            "Phosphorus concentration",
            "Nickel concentration",
            "Chromium concentration",
            "Molybdenum concentration",
            "Vanadium concentration",
            "Copper concentration",
            "Cobalt concentration",
            "Tungsten concentration",
            "Oxygen concentration",
            "Titanium concentration",
            "Nitrogen concentration",
            "Aluminium concentration",
            "Boron concentration",
            "Niobium concentration",
            "Tin concentration",
            "Arsenic concentration",
            "Antimony concentration",
            "Current",
            "Voltage",
            "AC or DC",
            "Electrode positive or negative",
            "Heat input",
            "Interpass temperature",
            "Type of weld",
            "Post weld heat treatment temperature",
            "Post weld heat treatment time",
            "Yield strength",
            "Ultimate tensile strength",
            "Elongation ",
            "Reduction of Area",
            "Charpy temperature",
            "Charpy impact toughness",
            "Hardness",
            "50 % FATT ",
            "Primary ferrite in microstructure",
            "Ferrite with second phase",
            "Acicular ferrite",
            "Martensite",
            "Ferrite with carbide aggreagate",
            "Weld ID"]



df = pd.read_csv("welddb.csv", names=colonnes, header=None)

"""
resultat = {}

for colonne in df.columns:
    non_n_count = df[df[colonne] != "N"].shape[0]
    resultat[colonne] = non_n_count

with open("resultat.txt", "w") as fichier_txt:
    json.dump(resultat, fichier_txt, indent=4)

print("Le fichier résultat.txt a été créé avec succès.")
 """

def est_nombre(valeur):
    try:
        float(valeur) 
        return True
    except ValueError:
        pass
    
    if isinstance(valeur, str) and re.match(r"<\d+(\.\d+)?", valeur):
        return True
    
    if valeur == "N":
        return True
    
    return False
ind = 0
""" for colonne in df.columns:
    toutes_nombres = df[colonne].apply(est_nombre).all()
    
    if toutes_nombres:
        print(f"La colonne '{colonne}' contient uniquement des nombres. {ind}")
    else:
        print(f"La colonne '{colonne}' ne contient PAS uniquement des nombres. {ind}")
    ind += 1 """

"""
def convertir_en_nombre(valeur):
    if isinstance(valeur, str) and re.match(r"<\d+(\.\d+)?", valeur):
        return float(valeur[1:])
    try:
        return float(valeur)
    except ValueError:
        return None

nb_non_N = df.apply(lambda col: col[col != "N"].count())

chiffres_ou_autre = df.apply(lambda col: "chiffres" if col.apply(est_nombre).all() else "autre")

moyenne_colonnes = df.apply(
    lambda col: col.apply(convertir_en_nombre).mean() if col.apply(est_nombre).all() else "N"
)

ecart_type_colonnes = df.apply(
    lambda col: col.apply(convertir_en_nombre).std() if col.apply(est_nombre).all() else "N"
)

resultat_df = pd.DataFrame([nb_non_N, chiffres_ou_autre, moyenne_colonnes, ecart_type_colonnes],
                           index=["Nb valeurs différentes de 'N'", "Chiffres ou Autre", "Moyenne", "Écart-type"])

resultat_df = resultat_df.T

resultat_df.to_csv("nouveau_fichier.csv", index_label="Colonnes")

print("Le fichier 'nouveau_fichier.csv' a été créé avec succès.") """

# print(df["Hardness"].tolist())

def repartition_valeurs(L):
    dico = {}
    for l in L:
        if l in dico:
            dico[l] += 1
        else:
            dico[l] = 1
    return dico

print(repartition_valeurs(df["Nitrogen concentration"].tolist()))
