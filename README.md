# Inibsa Centro de Gestión Automatizada (CGA)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-snnTorch-EE4C2C.svg)
![Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-F7DF1E.svg)
![Status](https://img.shields.io/badge/Status-Hackathon_MVP-success.svg)

Un sistema integral de apoyo a la decisión comercial diseñado para transformar la fuerza de ventas de Inibsa. Esta plataforma híbrida cruza predicciones de una red neuronal neuromórfica (SNN) con reglas de negocio estrictas para evitar fugas de clientes y capturar ventanas de oportunidad, organizando el trabajo en una agenda semanal inteligente.

---

## Características Principales

1. **Radar Predictivo (IA):** Anticipa las necesidades de compra de las clínicas a 7 días vista gracias a un modelo temporal basado en Spiking Neural Networks (SNN).
2. **Motor de Retención (Reglas de Negocio):** Identifica anomalías en el patrón de consumo, detectando desgaste de cuota de mercado o riesgo inminente de pérdida de clientes de alto valor.
3. **Distribución Inteligente:** Un dashboard interactivo que organiza las alertas críticas (urgencias máximas) en los primeros días de la semana laboral.
4. **Filtrado Avanzado:** Capacidad de segmentación cruzada por Región (ej. Madrid) y Familia de Producto (ej. Anestesia, Restauración) para un enfoque ultra-personalizado.

---

## Arquitectura y Tecnología

El proyecto está dividido en tres capas principales:

* **Backend Predictivo (IA):** * `PyTorch` y `snnTorch`: Arquitectura de neuronas LIF (Leaky Integrate-and-Fire) optimizada para la predicción de series temporales.
  * *Loss Function*: Implementación de un Weighted MSE para balancear la precisión y el recall penalizando los falsos negativos.
* **Procesamiento y Lógica:** * `Pandas` y `Scikit-learn`: Para el tratamiento del dataset histórico, agrupación en familias y escalado de variables.
* **Frontend y API:** * Interfaz *Single Page Application* (SPA) construida con **HTML, CSS y Vanilla JavaScript**.
  * Mini-servidor web (vía `Flask`) para la ingesta dinámica de datos en formato JSON. (no implementado en la Demo)

---

## Estructura del Proyecto

```text
inibsa-command-center/
├── app.py                  # Cerebro, genera un JSON con las señales
├── index.html              # Interfaz de usuario
├── requirements.txt        # Dependencias de Python
├── outputs/
│   └── signals.json        # Archivo consumido por el FrontEnd
├── data/
│   └── inibsa.png          # Logo jeje
├── snn.py                  # AI model
├── main.py                 # Handles the AI model inputs and outputs
├──data_anal.ipynb          # Initial testing with the dataset
```

# Instalación y Uso
## 1.Clonar el repositorio
```
git clone [https://github.com/Schebbair/InterHack.git](https://github.com/Schebbair/InterHack.git)
cd inibsa-command-center
```
## 2. Configurar el entorno virtual e instalar dependencias
Se recomienda utilizar un entorno virtual (venv o conda):
```
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## 3. Ejecutar la Interfaz (Modo Local)
Si ya dispones del archivo outputs/signals.json pre-generado, no necesitas levantar un servidor complejo. Simplemente abre el archivo HTML en tu navegador:

Navega a la carpeta del proyecto.

Haz doble clic en index.html o ábrelo en Google Chrome / Firefox.

(Nota para desarrollo: Si experimentas problemas de CORS al leer el JSON localmente mediante fetch, puedes lanzar un servidor de pruebas rápido con: python -m http.server 8000 y acceder a http://localhost:8000/index.html)

# Formato de Datos (API Contract)
El frontend espera leer un archivo signals.json con la siguiente estructura, permitiendo una fácil escalabilidad y conexión con futuros CRMs:
```
{
  "signals": [
    {
      "id": "signal-1234",
      "type": "type_4_promiscuous_restock",
      "clientId": "1000101643",
      "family": "Familia C1",
      "urgency": "ALERTA URGENTE",
      "message": "Cliente cerca de reabastecerse. Conviene contactar para evitar competencia.",
      "metadata": {
        "provincia": "Madrid"
      }
    }
  ]
}
```
# Equipo CGA
Desarrollado para la Hackathon Inibsa 2026.
Del dato a la acción comercial.
