def update_Database(t, d, file):
    """
    Este programa actualiza la lista de clientes promiscuos según un threshold t 
    (clientes que compran menos de t% de su potencial con Inibsa se consideran promiscuos).
    Considera los datos desde la fecha actual (d) hasta un año atrás.
    """

    import pandas as pd
    from datetime import datetime, timedelta

    # Llegeix dades
    vendes = pd.read_excel(file, sheet_name="Ventas")
    categories = pd.read_excel(file, sheet_name="Productos")
    potential = pd.read_excel(file, sheet_name="Potencial")

    # Convertir data primer
    vendes["Fecha"] = pd.to_datetime(vendes["Fecha"], errors="coerce")

    # Filtrar últim any
    limit_date = datetime.today() - timedelta(days=365)
    vendes = vendes[vendes["Fecha"] >= limit_date].copy()

    # Crear columna Any si la vols usar
    vendes["Any"] = vendes["Fecha"].dt.year

    # Normalitzar tipus clau (important)
    vendes["Id. Producto"] = vendes["Id. Producto"].astype(str)
    categories["Id.Prod"] = categories["Id.Prod"].astype(str)

    # Merge producte → categoria
    df = vendes.merge(
        categories,
        left_on="Id. Producto",
        right_on="Id.Prod",
        how="left"
    )

    # Agrupar vendes últim any
    resultat = df.groupby(
        ["Any", "Id. Cliente", "Categoria_H"]
    )["Valores_H"].sum().reset_index()

    # Merge amb potencial
    potential["Id. Cliente"] = potential["Id. Cliente"].astype(int)
    resultat["Id. Cliente"] = resultat["Id. Cliente"].astype(int)

    dataframe = resultat.merge(
        potential,
        on=["Id. Cliente", "Categoria_H"],
        how="left"
    )

    # Percentatge
    dataframe["prop"] = dataframe["Valores_H"] / dataframe["Potencial_H"]

    # Netegem valors invàlids
    clean_df = dataframe[
        (dataframe["Potencial_H"].notna()) &
        (dataframe["Potencial_H"] > 0)
    ].copy()

    df = clean_df.copy()
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna(subset=["prop", "Categoria_H"])

    # filtre global
    df_filtrat = df[df["prop"] < t]

    # 3 arrays per categoria
    cat_clients = (
        df_filtrat.groupby("Categoria_H")["Id. Cliente"]
        .unique()
        .to_dict()
    )
    #print (cat_clients)
    #sizes = {cat: len(clients) for cat, clients in cat_clients.items()}
    #print(sizes)
    import pandas as pd

    rows = []

    for categoria, clients in cat_clients.items():
        for c in clients:
            rows.append({
                "Categoria_H": categoria,
                "Id. Cliente": c
            })


    dataset = pd.DataFrame(rows)

    # Escriure a Excel en nova pestanya
    with pd.ExcelWriter(
        "RawData.xlsx",
        mode="a",
        engine="openpyxl",
        if_sheet_exists="replace"
    ) as writer:
        dataset.to_excel(writer, sheet_name="Promiscuos", index=False)