# Redes Neuronales 2026

Trabajos prácticos de la materia, sobre clasificación de texto con el dataset **20 Newsgroups** (20 clases, TF-IDF).

| TP | Carpeta | Contenido |
|---|---|---|
| TP1 | `TP1_Softmax/` | Regresión softmax en PyTorch. Preprocesamiento con NLTK (stopwords, stemming/lematización), TF-IDF, búsqueda de hiperparámetros con Optuna (15 trials), análisis de dinámica, tiempos de convergencia, bias/varianza y preguntas de reflexión |
| TP2 | `TP2_MLP/` | MLP (activaciones, profundidad, ancho, optimizadores y sus momentos) con Optuna y pruning, estudios por optimizador, rangos de búsqueda, overfitting, regularización (dropout / weight decay / max-norm) y comparación contra softmax con presupuestos iguales |

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Los recursos de NLTK (`stopwords`, `wordnet`, `punkt`) se descargan solos al ejecutar la primera celda de cada notebook, y el dataset lo baja `scikit-learn` la primera vez (`fetch_20newsgroups`).

## Cómo ejecutar

Abrir el notebook de cada carpeta y ejecutar **Run All**. Todo se ejecuta en CPU (los tiempos son relativos a esa máquina). Los notebooks escriben en la carpeta desde la que se abren:

- `figures/` y `results/`: gráficos y tablas (CSV) generados.
- `runs/`: logs de **TensorBoard** → `tensorboard --logdir runs`
- `mlflow.db`: registro de **MLflow** (parámetros, métricas por época y tiempos de cada trial y de cada corrida) → `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- `mlp_study.db` (solo TP2): base SQLite de los estudios de Optuna.

### Notas

- Las celdas del TP2 que crean estudios de Optuna son **idempotentes**: si `mlp_study.db` ya existe, completan hasta el número de trials objetivo en vez de agregar más (y marcan como `FAIL` cualquier trial que haya quedado colgado por una interrupción). Para empezar de cero, borrar el `.db`.
- Las bases de datos (`*.db`), `runs/`, `mlruns/`, `figures/` y `results/` **no se versionan** (ver `.gitignore`): se regeneran ejecutando los notebooks.
- Semilla fija (`SEED = 42`) en todo. Los resultados pueden variar levemente entre máquinas (CPU/versión de PyTorch).
- El conjunto de **test** se usa una sola vez por TP para la evaluación final; todas las decisiones (hiperparámetros, número de épocas) se toman con un split de validación 80/20 estratificado.
