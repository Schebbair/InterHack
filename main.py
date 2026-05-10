import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

#from snn import SNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BACK_DAYS = 60
FORWARD_DAYS = 7
NUM_EPOCHS = 25

def weighted_mse_loss(pred, target, penalty=15.0):
    se = (pred - target) ** 2
    weights = torch.where(target > 0.0, penalty, 1.0)
    loss_final = torch.mean(weights * se)
    return loss_final

df_potencial = pd.read_csv('data/Potencial.csv', sep=',', decimal=',', thousands='.')
df_clientes = pd.read_csv('data/Clientes.csv', sep=',')
df_productos = pd.read_csv('data/Productos.csv', sep=',')
df_ventas = pd.read_csv('data/Ventas.csv', sep=',', decimal=',', thousands='.', dtype={'Num.Fact': str})
df_campanas = pd.read_csv('data/Campañas.csv', sep=',')

#columna codigo postal tu sabe
df_clientes.rename(columns={'Unnamed: 1': 'CodigoPostal'}, inplace=True)

df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'], errors='coerce')
df_campanas['Fecha inicio'] = pd.to_datetime(df_campanas['Fecha inicio'], errors='coerce')
df_campanas['Fecha fin'] = pd.to_datetime(df_campanas['Fecha fin'], errors='coerce')

df_ventas = df_ventas.merge(df_productos, left_on='Id.Producto', right_on='Id.Prod', how='left')
df_ventas = df_ventas.merge(df_clientes, on='Id.Cliente', how='left')

ventas_reales = df_ventas[df_ventas['Unidades'] > 0].copy()

family = df_productos['Familia'].dropna().unique()
family_idx = {f: i for i, f in enumerate(family)}
ventas_reales['Family_idx'] = ventas_reales['Familia'].map(family_idx)

N_FAMILIES = len(family)
X_global = []
Y_global = []

compras = ventas_reales.groupby('Id.Cliente').size()
clientes_validos = compras[compras > 10].index

clientes_validos = clientes_validos[:50]

fecha_min = ventas_reales['Fecha'].min()
fecha_max = ventas_reales['Fecha'].max()
calendar = pd.date_range(start=fecha_min, end=fecha_max, freq='D')

for cliente in clientes_validos:
    df_cliente = ventas_reales[ventas_reales['Id.Cliente'] == cliente]

    df_pivot = df_cliente.pivot_table(index='Fecha', columns='Family_idx', values='Unidades', aggfunc='sum', fill_value=0)
    df_dense = df_pivot.reindex(calendar, fill_value=0)

    for i in range(N_FAMILIES):
        if i not in df_dense.columns:
            df_dense[i] = 0
    df_dense = df_dense[range(N_FAMILIES)]
    df_dense['Mes_Norm'] = df_dense.index.month / 12.0


    Y_matrix = df_dense[range(N_FAMILIES)].values
    X_matrix = df_dense.values
    X_matrix[:, :N_FAMILIES] = (X_matrix[:, :N_FAMILIES] > 0).astype(float) * 2.0

    for i in range(len(calendar) - BACK_DAYS - FORWARD_DAYS):

        american_history_x = X_matrix[i : i + BACK_DAYS]

        future_y = Y_matrix[i + BACK_DAYS : i + BACK_DAYS + FORWARD_DAYS]
        target_y = future_y.sum(axis=0)

        X_global.append(american_history_x)
        Y_global.append(target_y)

print("Total muestras:", len(X_global))

X_tensor = torch.tensor(np.array(X_global), dtype=torch.float32)
Y = np.array(Y_global)

scaler = MinMaxScaler(feature_range=(0, 1))
Y_norm = scaler.fit_transform(Y)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

X_tensor = X_tensor.to(device)
Y_tensor = Y_tensor.to(device)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

model = SNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0

    for x_batch, y_batch in dataloader:
        
        optimizer.zero_grad()

        #Pytorch tu sabe
        x_batch_snn = x_batch.permute(1, 0, 2)
        pred = model(x_batch_snn)
        loss = weighted_mse_loss(pred, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss {total_loss/len(dataloader):.4f}")


fecha_inicio = fecha_max - pd.Timedelta(days=BACK_DAYS - 1)
calendar_inferencia = pd.date_range(start=fecha_inicio, end=fecha_max, freq='D')

cliente_to_pred = clientes_validos[0]
df_cliente_inferencia = ventas_reales[ventas_reales['Id.Cliente'] == cliente_to_pred]

df_pivot_inf = df_cliente_inferencia.pivot_table(index='Fecha', columns='Family_idx',values='Unidades', aggfunc='sum', fill_value=0 )
df_dense_inf = df_pivot_inf.reindex(calendar_inferencia, fill_value=0)

for i in range(N_FAMILIES):
    if i not in df_dense_inf.columns:
        df_dense_inf[i] = 0

df_dense_inf = df_dense_inf[range(N_FAMILIES)]
df_dense_inf['Mes_Norm'] = df_dense_inf.index.month / 12.0

X_matrix_inf = df_dense_inf.values
X_matrix_inf[:, :N_FAMILIES] = (X_matrix_inf[:, :N_FAMILIES] > 0).astype(float) * 2.0
x_tensor_new = torch.tensor(X_matrix_inf, dtype=torch.float32).unsqueeze(0)

x_tensor_new = x_tensor_new.to(device)

x_batch_new = x_tensor_new.permute(1,0,2)

print("\n---------EVAL-----------\n")
model.eval()
total_pred = []
total_target = []

with torch.no_grad():

    for x_batch, y_batch in dataloader:

        x_batch_snn = x_batch.permute(1, 0, 2)
        pred_norm = model(x_batch_snn)
        
        total_pred.append(pred_norm.cpu().numpy())
        total_target.append(y_batch.cpu().numpy())

preds_array = np.concatenate(total_pred, axis=0)
targets_array = np.concatenate(total_target, axis=0)

preds_real = scaler.inverse_transform(preds_array)
targets_real = scaler.inverse_transform(targets_array)

preds_real = np.maximum(0, preds_real)

mae = mean_absolute_error(targets_real, preds_real)
print(f"Error Medio Absoluto (MAE): Nos equivocamos en {mae:.2f} unidades de media por producto.")

compras_reales_binarias = (targets_real > 0).astype(int)
compras_reales_positivas = compras_reales_binarias == 1

compras_pred_binarias_prueba = (preds_real > 2.0).astype(int)
    
# Precisión
aciertos_prueba = (compras_reales_binarias == compras_pred_binarias_prueba).mean()
    
# Recall
aciertos_pos_prueba = compras_pred_binarias_prueba[compras_reales_positivas] == 1
recall_prueba = aciertos_pos_prueba.mean()
    
print(f"Precisión: {aciertos_prueba*100:.1f}% | Recall: {recall_prueba*100:.1f}%")