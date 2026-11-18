# Redes Neuronales 2026

Trabajos prácticos de la materia sobre clasificación de texto con el dataset **20 Newsgroups** (20 clases, TF-IDF), en PyTorch.

| TP | Carpeta | Contenido |
|---|---|---|
| TP1 | `TP1_Softmax/` | Regresión softmax. Preprocesamiento con NLTK (stopwords, stemming/lematización), TF-IDF, búsqueda de hiperparámetros con Optuna (15 trials), dinámica de entrenamiento, tiempos de convergencia, bias/varianza y preguntas de reflexión |
| TP2 | `TP2_MLP/` | MLP (activaciones, profundidad, ancho, optimizadores y momentos) con Optuna y pruning, estudios por optimizador, rangos de búsqueda, overfitting, dropout / weight decay / max-norm y comparación con softmax con presupuestos iguales |
| TP3 | `TP3_Regularizacion/` | Regularización como hiperparámetros: **L1 / L2 / L1+L2, Dropout y BatchNorm**. Búsqueda de Optuna con y sin regularización (mismo presupuesto), experimentos controlados (un factor por vez) midiendo **valores absolutos de pesos, tiempos, learning rate y ruido en las curvas**, y justificación con lo visto en clase |

## Instalación

Requiere **Python 3.11 a 3.13** (en versiones más nuevas pueden faltar paquetes precompilados).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Los recursos de NLTK se descargan solos en la primera celda y el dataset lo baja `scikit-learn` la primera vez (`fetch_20newsgroups`). En Windows, `torch` puede pedir el *Microsoft Visual C++ Redistributable*.

## Cómo ejecutar

Abrir el notebook de la carpeta del TP, elegir el kernel del entorno donde se instalaron los requisitos y ejecutar **Run All**. Todo corre en CPU (los tiempos son relativos a la máquina). Duración aproximada: TP1 ~15 min, TP2 ~2 h, TP3 ~1.5–2 h (con `N_TRIALS = 30`; se puede bajar en la primera celda).

Cada notebook escribe en la carpeta desde la que se ejecuta:

- `figures/` y `results/`: gráficos y tablas (CSV) generados.
- `runs/`: logs de **TensorBoard** → `tensorboard --logdir runs`
- `mlflow.db`: registro de **MLflow** → `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- `mlp_study.db` (TP2) y `tp3_study.db` (TP3): estudios de **Optuna** en SQLite.
- `results/tp3_corridas.jsonl` (TP3): caché de las corridas controladas.

### Notas

- **Se pueden retomar:** los estudios de Optuna de TP2 y TP3 son idempotentes (completan hasta el número de trials objetivo y marcan como `FAIL` los trials que quedaron colgados por una interrupción), y el TP3 no repite las corridas controladas ya guardadas. Para empezar de cero, borrar los `.db` y `results/tp3_corridas.jsonl`.
- **Qué se versiona:** el `.gitignore` es una *whitelist*: solo suben los notebooks, el README y `requirements.txt`. Bases de datos, `runs/`, `mlruns/`, `figures/` y `results/` se regeneran ejecutando los notebooks.
- Semilla fija (`SEED = 42`); los resultados pueden variar levemente entre máquinas.
- El **test** se usa una sola vez por modelo y semilla; todas las decisiones (hiperparámetros, número de épocas) se toman con un split de validación 80/20 estratificado.
- Las conclusiones de cada notebook se **generan a partir de los resultados** de la última ejecución, para que texto y números coincidan.
