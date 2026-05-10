def generar_alertes_individuals(id_user, id_producto, familia, valor_predecido, valor_real):
    # Alerta de Compliment (Control)
    # Calcul de la diferencia relativa
    if valor_predecido != 0:
        dif_relativa = abs(valor_real - valor_predecido) / valor_predecido
    else:
        dif_relativa = 0

    # Determinacion de la urgencia
    if dif_relativa > 0.50:
        urgencia = "URGENTE"
    else:
        urgencia = ""

    return (
        f"ALERTA {urgencia}: El control de prediccion para el cliente {id_user} "
        f"ha fallado para el producto {id_producto} (familia: {familia}) "
        f"con una diferencia relativa de: {dif_relativa:.2%}"
    )
