# Redes Neuronales 2026

Trabajos prácticos de la materia, en PyTorch: clasificación de texto con el dataset **20 Newsgroups** (TP1 a TP3, TF-IDF) y de imágenes de lesiones de piel con el dataset **ISIC** (TP4, redes convolucionales).

| TP | Carpeta | Contenido |
|---|---|---|
| TP1 | `TP1_Softmax/` | Regresión softmax. Preprocesamiento con NLTK (stopwords, stemming/lematización), TF-IDF, búsqueda de hiperparámetros con Optuna (15 trials), dinámica de entrenamiento, tiempos de convergencia, bias/varianza y preguntas de reflexión |
| TP2 | `TP2_MLP/` | MLP (activaciones, profundidad, ancho, optimizadores y momentos) con Optuna y pruning, estudios por optimizador, rangos de búsqueda, overfitting, dropout / weight decay / max-norm y comparación con softmax con presupuestos iguales |
| TP3 | `TP3_Regularizacion/` | Regularización como hiperparámetros: **L1 / L2 / L1+L2, Dropout y BatchNorm**. Búsqueda de Optuna con y sin regularización (mismo presupuesto), experimentos controlados (un factor por vez) midiendo **valores absolutos de pesos, tiempos, learning rate y ruido en las curvas**, y justificación con lo visto en clase |
| TP4 | `TP4_CNN/` | **Redes convolucionales** sobre imágenes dermatoscópicas (ISIC, 9 clases, ~880 imágenes). EDA (desbalance, resoluciones, color, duplicados), eliminación de duplicados y split estratificado 60/20/20, augmentation a mano, MLP vs CNN (parámetros, tiempos, robustez a traslaciones), arquitectura (profundidad, kernel, filtros, resolución, campo receptivo), overfitting y augmentation, evaluación final en test (una sola vez), filtros y mapas de activación, y controles de sesgo y fuga de información |

## Instalación

Requiere **Python 3.11 a 3.13** (en versiones más nuevas pueden faltar paquetes precompilados).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

En TP1–TP3 los recursos de NLTK se descargan solos en la primera celda y el dataset lo baja `scikit-learn` la primera vez (`fetch_20newsgroups`). En Windows, `torch` puede pedir el *Microsoft Visual C++ Redistributable*.

## Cómo ejecutar

Abrir el notebook de la carpeta del TP, elegir el kernel del entorno donde se instalaron los requisitos y ejecutar **Run All**. Todo corre en CPU (los tiempos son relativos a la máquina). Duración aproximada: TP1 ~15 min, TP2 ~2 h, TP3 ~1.5–2 h (con `N_TRIALS = 30`; se puede bajar en la primera celda), TP4 ~20–25 min.

### TP4: datos

El dataset **no se sube al repositorio**. Descargarlo del repositorio de la cátedra (<https://github.com/cselmo/CNN-en-Skin-Dataset>, carpeta `data/Split_smol`, con `train/` y `val/`) y dejarlo en `TP4_CNN/data/Split_smol/`. El notebook se ejecuta **desde la carpeta `TP4_CNN/`** (la ruta `data/Split_smol` es relativa). Las constantes de la sección 0 (semillas, épocas, tamaños) no se modificaron, y la semilla del split (`SEED_SPLIT = 42`) no debe cambiarse. Los cambios respecto del notebook base están en la tabla del principio del notebook.

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
- Las conclusiones de cada notebook se **generan a partir de los resultados** de la última ejecución, para que texto y números coincidan (en el TP4, las celdas de análisis construyen el texto desde las variables de la corrida; las interpretaciones cualitativas conviene releerlas si se cambia alguna constante).
- TP4: la normalización usa la media y el desvío por canal **solo de train**, la augmentation se aplica **solo a train y después del split**, y el test se usa **una sola vez** (sección 8).
