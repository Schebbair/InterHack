import numpy as np

# pasame los ids para poder hacer el mensaje que se envia al frontend
def SPC(lista_errores, id_cliente, id_producto):
    """
    Evalúa si un cliente está abandonando un producto basándose en el análisis SPC.
    
    Parámetros:
    lista_errores (list o array): Lista cronológica de errores (Real - Predicción) en UNIDADES.
                                  El último elemento de la lista corresponde al día de hoy/semana actual.
    id_cliente (str o int): Identificador del cliente.
    id_producto (str o int): Identificador del producto.
                                  
    Retorna:
    String con el mensaje de alerta si hay riesgo de pérdida.
    None (null) si el consumo está bajo control o hay un repunte de demanda.
    """
    
    # Necesitamos al menos 7 datos para poder evaluar la regla de tendencia
    if len(lista_errores) < 7:
        return None
        
    # Separamos el histórico (el pasado) del valor actual (hoy)
    historico = lista_errores[:-1]
    error_hoy = lista_errores[-1]
    
    # Calculamos Línea Central (Media) y Desviación Estándar
    media = np.mean(historico)
    std = np.std(historico, ddof=1) # ddof=1 es lo estadísticamente correcto para muestras
    
    # CRÍTICO: Si el cliente siempre tiene un error idéntico, la desviación será 0.
    # Al trabajar en unidades, fijamos un mínimo de "1 unidad de ruido" para evitar falsos positivos.
    if std < 1.0:
        std = 1.0
        
    # Calculamos únicamente el Límite de Control Inferior (LCL)
    limite_inferior = media - (3 * std)
    
    # --- REGLA 1: Outlier por caída brusca ---
    # El error de hoy es negativo y ha perforado el límite inferior tolerado
    if error_hoy < limite_inferior:
        return f"ALERTA URGENTE: Estamos perdiendo al cliente {id_cliente} en el producto {id_producto} (Caída brusca por debajo del límite)."
        
    # --- REGLA 2: Tendencia silenciosa a la baja (Reglas de Nelson) ---
    # Revisamos si los últimos 7 errores (incluyendo el de hoy) están TODOS por debajo de la media histórica
    ultimos_7_errores = lista_errores[-7:]
    if all(error < media for error in ultimos_7_errores):
        return f"ALERTA: Estamos perdiendo al cliente {id_cliente} en el producto {id_producto} (Desgaste lento: 7 periodos seguidos por debajo de lo normal)."
        
    # Si no salta ninguna alerta (es decir, proceso estable o repunte de compras)
    return None


def SCP_familia(agregado_lista_errores, id_producto):
    """
    Evalúa si un producto está perdiendo cuota basándose en el análisis SPC agregado.
    
    Parámetros:
    agregado_lista_errores (list o array): Lista cronológica de errores (Real - Predicción) en UNIDADES.
                                  El último elemento de la lista corresponde al día de hoy/semana actual.
    id_producto (str o int): Identificador del producto.
                                  
    Retorna:
    String con el mensaje de alerta si hay riesgo de pérdida de cuota.
    None (null) si el consumo está bajo control o hay un repunte de demanda.
    """
    
    # Necesitamos al menos 7 datos para poder evaluar la regla de tendencia
    if len(agregado_lista_errores) < 7:
        return None
        
    # Separamos el histórico (el pasado) del valor actual (hoy)
    historico = agregado_lista_errores[:-1]
    error_hoy = agregado_lista_errores[-1]
    
    # Calculamos Línea Central (Media) y Desviación Estándar
    media = np.mean(historico)
    std = np.std(historico, ddof=1)
    
    # CRÍTICO: Suelo de seguridad Fijo (Hardcoded).
    # Ajusta este número (ej. 5, 10, 50) según lo que consideres "ruido irrelevante" 
    # para un producto a nivel global en tu empresa.
    ruido_minimo_fijo = 10.0
    
    if std < ruido_minimo_fijo:
        std = ruido_minimo_fijo
        
    # Calculamos únicamente el Límite de Control Inferior (LCL)
    limite_inferior = media - (3 * std)
    
    # --- REGLA 1: Outlier por caída brusca ---
    # El error de hoy es negativo y ha perforado el límite inferior tolerado
    if error_hoy < limite_inferior:
        return f"ALERTA URGENTE: Estamos perdiendo cuota en el producto {id_producto} (Caída brusca por debajo del límite)."
        
    # --- REGLA 2: Tendencia silenciosa a la baja (Reglas de Nelson) ---
    # Revisamos si los últimos 7 errores (incluyendo el de hoy) están TODOS por debajo de la media histórica
    ultimos_7_errores = agregado_lista_errores[-7:]
    if all(error < media for error in ultimos_7_errores):
        return f"ALERTA: Estamos perdiendo cuota en el producto {id_producto} (Desgaste lento: 7 periodos seguidos por debajo de lo normal)."
        
    # Si no salta ninguna alerta
    return None