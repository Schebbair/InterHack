import pandas as pd

"""
Usando este programa se genera una hoja excel con los potenciales erroneos 
(mas de 100% de consumo) corregidos. Estos son útiles para actualizar y corregir 
los valores de potencial de la database a medida que el programario funciona y la demanda varia.
"""

# Llegir Excel amb diverses fulles
file = "RawData.xlsx"

vendes = pd.read_excel(file, sheet_name="Ventas")
categories = pd.read_excel(file, sheet_name="Productos")

# Convertir data
vendes["Fecha"] = pd.to_datetime(vendes["Fecha"])

# Crear columna any
vendes["Any"] = vendes["Fecha"].dt.year

# Unir taules (relació producte → categoria)
df = vendes.merge(
    categories,
    left_on="Id. Producto",
    right_on="Id.Prod",
    how="left"
)

# Agrupar resultat
resultat = df.groupby(
    ["Any", "Id. Cliente", "Categoria_H"]
)["Valores_H"].sum().reset_index()

resultat = resultat.sort_values(
    by=["Id. Cliente", "Any", "Categoria_H"]
).reset_index(drop=True)

potential = pd.read_excel(file, sheet_name="Potencial")
potential["Id. Cliente"] = potential["Id. Cliente"].astype(int)
dataframe = resultat.merge(
    potential,
    on=["Id. Cliente", "Categoria_H"],
    how="left"
)

dataframe["prop"] = dataframe["Valores_H"] / dataframe["Potencial_H"]

# Remove rows where potential is 0 or missing
clean_df = dataframe[
    (dataframe["Potencial_H"] > 0) &
    (dataframe["Potencial_H"].notna())
].copy()

# Keep only proportion > 1
clean_df = clean_df[clean_df["prop"] > 1].copy()
max_prop = clean_df.loc[
    clean_df.groupby(
        ["Id. Cliente", "Categoria_H"]
    )["prop"].idxmax()
].copy()
max_prop["Updated_Potential"] = (
    max_prop["Potencial_H"] * max_prop["prop"]
)

# Final output
output = max_prop[[
    "Id. Cliente",
    "Categoria_H",
    "prop",
    "Updated_Potential"
]]

mapping = {
    "Categoria C1": "Anestesia",
    "Categoria C2": "Bioseguridad",
    "Categoria T1": "Biomateriales"
}

output["Familia"] = output["Categoria_H"].map(mapping)
output = output[
    [
        "Id. Cliente",
        "Familia",
        "Categoria_H",
        "Updated_Potential"
    ]
]

output.to_excel("output.xlsx", index=False)