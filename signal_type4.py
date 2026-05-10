import pandas as pd
import datetime


P_weights = {
"categoria_c1" : {
    1: 0.0127, 2: 0.0218, 3: 0.0219, 4: 0.0198, 5: 0.0219, 6: 0.0216,
    7: 0.0218, 8: 0.0207, 9: 0.0225, 10: 0.0188, 11: 0.0190, 12: 0.0296,
    13: 0.0205, 14: 0.0203, 15: 0.0192, 16: 0.0167, 17: 0.0219, 18: 0.0162,
    19: 0.0213, 20: 0.0195, 21: 0.0214, 22: 0.0204, 23: 0.0281, 24: 0.0203,
    25: 0.0176, 26: 0.0193, 27: 0.0211, 28: 0.0174, 29: 0.0154, 30: 0.0143,
    31: 0.0110, 32: 0.0057, 33: 0.0030, 34: 0.0062, 35: 0.0165, 36: 0.0269,
    37: 0.0235, 38: 0.0206, 39: 0.0220, 40: 0.0235, 41: 0.0185, 42: 0.0212,
    43: 0.0218, 44: 0.0183, 45: 0.0255, 46: 0.0211, 47: 0.0230, 48: 0.0258,
    49: 0.0194, 50: 0.0207, 51: 0.0163, 52: 0.0065
},
"categoria_c2" : {
    1: 0.0126, 2: 0.0257, 3: 0.0258, 4: 0.0243, 5: 0.0245, 6: 0.0192,
    7: 0.0206, 8: 0.0177, 9: 0.0240, 10: 0.0184, 11: 0.0169, 12: 0.0281,
    13: 0.0199, 14: 0.0195, 15: 0.0192, 16: 0.0172, 17: 0.0226, 18: 0.0149,
    19: 0.0235, 20: 0.0196, 21: 0.0221, 22: 0.0188, 23: 0.0260, 24: 0.0222,
    25: 0.0161, 26: 0.0230, 27: 0.0213, 28: 0.0189, 29: 0.0168, 30: 0.0152,
    31: 0.0090, 32: 0.0054, 33: 0.0023, 34: 0.0045, 35: 0.0161, 36: 0.0245,
    37: 0.0237, 38: 0.0220, 39: 0.0249, 40: 0.0225, 41: 0.0183, 42: 0.0212,
    43: 0.0186, 44: 0.0170, 45: 0.0248, 46: 0.0202, 47: 0.0215, 48: 0.0253,
    49: 0.0199, 50: 0.0202, 51: 0.0147, 52: 0.0088
},
"categoria_t1" : {
    1: 0.0109, 2: 0.0204, 3: 0.0250, 4: 0.0249, 5: 0.0238, 6: 0.0241,
    7: 0.0230, 8: 0.0226, 9: 0.0225, 10: 0.0192, 11: 0.0235, 12: 0.0280,
    13: 0.0185, 14: 0.0205, 15: 0.0195, 16: 0.0211, 17: 0.0201, 18: 0.0172,
    19: 0.0243, 20: 0.0215, 21: 0.0223, 22: 0.0199, 23: 0.0246, 24: 0.0204,
    25: 0.0183, 26: 0.0200, 27: 0.0205, 28: 0.0176, 29: 0.0151, 30: 0.0108,
    31: 0.0068, 32: 0.0028, 33: 0.0021, 34: 0.0042, 35: 0.0131, 36: 0.0210,
    37: 0.0211, 38: 0.0233, 39: 0.0238, 40: 0.0254, 41: 0.0187, 42: 0.0237,
    43: 0.0244, 44: 0.0178, 45: 0.0245, 46: 0.0255, 47: 0.0231, 48: 0.0257,
    49: 0.0166, 50: 0.0152, 51: 0.0156, 52: 0.0056
}
}

# Función para simular el consumo diario y calcular cuándo se agotaría el saldo de la compra para avisar de posible contacto
# si el cliente tiene un expected buy time mayour a exhaust
# asumo que encuentras el potencial y p_weights según la categoría del cliente previo a la funcion
# inputs: fecha de la compra, valor de la compra, potencial anual del cliente, perfil estacional (p_weights)
def days_until_exhaust(transaction_date, val_trans, potential, p_weights):
    current_date = pd.to_datetime(transaction_date, format='%d/%m/%Y')
    balance = val_trans
    days_passed = 0
    
    # Bucle de simulación: corremos día a día hasta que el saldo sea 0 o negativo
    while balance > 0:
        # 1. ¿En qué semana del año estamos hoy en la simulación?
        semana_actual = current_date.isocalendar().week
        
        # 2. ¿Cuánto gasta el cliente en TODA esta semana según su potencial?
        # Usamos .get(semana, promedio) por si hay alguna semana sin datos históricos
        peso_semana = p_weights.get(semana_actual, 1/52) 
        gasto_esta_semana = potential * peso_semana
        
        # 3. ¿Cuánto gasta HOY? (Gasto semanal / 7)
        gasto_diario = gasto_esta_semana / 7.0
        
        # 4. Restamos el consumo de hoy al saldo de la compra
        balance -= gasto_diario
        
        # 5. Avanzamos al día siguiente
        current_date += pd.Timedelta(days=1)
        days_passed += 1
        
        # Seguridad: Si un cliente hace una compra masiva superior a 2 años de potencial, cortamos
        if days_passed > 730: 
            break
    # Formato de return %y-%m-%d 2024-03-12 00:00:00
    return current_date # Esta es la fecha en la que el saldo llega a 0, es una alerta URGENTE en la semana anterior