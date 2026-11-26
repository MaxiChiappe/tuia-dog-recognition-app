# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Pipeline completo de Computer Vision sobre el **70 Dog Breeds Image Dataset**:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

**Equipo:** CHIAPPE, Maximiliano (C-7542/6) · LENARDUZZI, Juan (L-3398/7)

## Inicio rápido (para probar la entrega)

1. Descargar el dataset en `data/dataset` (ver [Dataset](#dataset)).
2. Descargar los checkpoints entrenados y colocarlos en `models/`:
   - **Link (solo lectura):** https://drive.google.com/drive/folders/1WnbeiZCLiiL_dbsBIPEEw1tqZpgT149K?usp=sharing
   - `models/resnet18_finetuned.pth` (Modelo A) y `models/cnn_custom.pth` (Modelo B).
3. Construir y levantar la aplicación:

   ```bash
   docker compose build
   docker compose up -d
   ```

4. Indexar la galería (base vectorial con los embeddings `baseline`, el modelo elegido para la búsqueda):

   ```bash
   docker compose exec backend python /scripts/build_index.py --split train
   ```

5. Abrir el frontend en `http://localhost:8080`. `http://localhost:8000/health` debe mostrar `embedding_model: baseline` y los dos checkpoints en `true`.

La configuración usada es la de `.env.docker.example`: no hace falta editarla.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset, EDA, filtrado, conjunto independiente y splits (ejecutada)
├── 2-train.ipynb                # Etapa 1 (NDCG@10), Etapa 2 (modelos A y B), validación de YOLO y comparación (ejecutada)
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env (incluye los hiperparámetros de entrenamiento)
│       ├── preprocessing.py     # [grupo] Preprocesamiento compartido (app y entrenamiento) y data augmentation
│       ├── training.py          # [grupo] Dataset, carga de splits, configuración y loop de entrenamiento
│       ├── models/              # [grupo] BreedClassifier (ResNet18 + cabeza) y CNN propia
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # Etapa 1 (implementada)
│       │   ├── classifier_service.py   # Etapa 2 (implementada)
│       │   ├── detection_service.py    # Etapa 3 (implementada)
│       │   └── task_manager.py
│       └── storage/             # Base vectorial en JSON o PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   ├── build_independent_set.py # [grupo] Arma el conjunto independiente (Oxford-IIIT Pet + Stanford Dogs)
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # Dataset y conjunto independiente, no se versionan
│   └── splits.csv               # [grupo] Splits finales train/valid/test tras el filtrado
├── models/                      # Checkpoints entrenados (.pth), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (opcional, para descargar el dataset por script)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Si no tienen [credenciales de Kaggle](https://www.kaggle.com/docs/api), descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

**Conjunto independiente** (solo para reproducir las notebooks, la app no lo necesita): [link de solo lectura](https://drive.google.com/drive/folders/15gcmPbUfTUqzgQ1kacKKCoGZUJj_8q2N?usp=sharing). Va en `data/dataset/independent/`. También se puede regenerar con `python scripts/build_independent_set.py`.

## Modelos

Los checkpoints entrenados no se versionan. Descárguenlos de https://drive.google.com/drive/folders/1WnbeiZCLiiL_dbsBIPEEw1tqZpgT149K?usp=sharing y colóquenlos en `models/`:

| Archivo | Modelo |
|---|---|
| `models/resnet18_finetuned.pth` | Modelo A: ResNet18 fine-tuned (estrategia `layer4`) |
| `models/cnn_custom.pth` | Modelo B: CNN propia |

Son modelos completos de PyTorch (`torch.save`), guardados en CPU: la app los carga sin GPU. El detector YOLOv8n se descarga solo en el primer uso.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan con un mensaje claro hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná el repositorio

```bash
git clone <url-del-fork>
cd tuia-dog-recognition-app
git checkout feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Copiá `.env.local.example` a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Armar el conjunto independiente (descarga Oxford-IIIT Pet y Stanford Dogs, ~1.5 GB)
python scripts/build_independent_set.py

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

Todo es configurable por `.env` (`src/lib/config.py`). Valores elegidos en `.env.docker.example` y `.env.local.example`:

| Variable | Valor | Por qué |
|---|---|---|
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `baseline` / `512` | Más robusto fuera de dominio que los embeddings entrenados (ver [Resultados](#resultados)) |
| `SIMILARITY_METRIC` | `cosine` | Con embeddings normalizados L2, cosine y l2 dan el mismo ranking |
| `SIMILARITY_THRESHOLD` | `0.82` | Calibrado con `valid`: percentil 1 del score top-1 (rechaza los 10 gatos de prueba) |
| `TOP_K` | `10` | La accuracy casi no cambia entre K=3 y K=20 |
| `YOLO_CONF_THRESHOLD` | `0.25` | Mejor equilibrio entre perros perdidos y cajas duplicadas (ver `2-train.ipynb`, sección 12) |
| `TRAIN_*`, `RESNET_FINETUNE`, `SPLITS_FILE` | ver archivos | Hiperparámetros de entrenamiento de la Etapa 2 y splits finales |

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Carga de checkpoints desde `models/`, persistencia en JSON o PostgreSQL + pgvector, métricas y visualización.
- Frontend Gradio con una pestaña por etapa.

### Implementado por el grupo

| Etapa | Función | Implementación |
|-------|---------|----------------|
| 1 | `extract_embedding` | ResNet18 ImageNet sin la capa final (`fc = Identity`), 512 dimensiones, normalizado L2. Preprocesamiento compartido (`preprocessing.py`): BGR→RGB, resize a 224, normalización de ImageNet. El modelo se carga una sola vez. |
| 1 | `search_similar_images` | `store.search` (pgvector, operador `<=>`) o recorrido de `store.all()` con `self.similarity`; Top-K ordenado por score. |
| 1 | `predict_breed_from_neighbors` | Voto ponderado por score; devuelve la similitud media de la raza ganadora. `unknown` si el mejor score queda bajo `SIMILARITY_THRESHOLD`. |
| 2 | `train_classifier` | Splits de `data/splits.csv`, augmentation solo en train, AdamW + coseno, label smoothing, early stopping por accuracy de valid; guarda el mejor checkpoint (en CPU) y el historial en `output/`. GPU automática si está disponible. |
| 2 | `evaluate_classifier` | Accuracy y precision, recall, specificity y F1 macro sobre test (con `metrics.py`); guarda las predicciones en CSV. |
| 2 | `extract_custom_embedding` | Penúltima capa del modelo activo (`model.embed`), normalizada L2, con validación de `EMBEDDING_DIM`. |
| 3 | `detect_dogs` | YOLOv8n preentrenado en COCO, filtrado por la clase `dog` y `YOLO_CONF_THRESHOLD`; cajas en píxeles enteros. |
| 3 | `classify_detected_dog` | Mismo preprocesamiento que el entrenamiento + softmax del modelo activo; las razas viajan dentro del checkpoint. |

## Dataset

Detalle completo en `1-pre-processing.ipynb`.

- **Corrección de etiquetas:** "American  Spaniel" (doble espacio, en `valid`) aparecía como una raza distinta: 71 → **70 razas**.
- **Filtrado (147 imágenes descartadas):** duplicados detectados con dHash (73 de train con copia en valid/test, 6 de valid con copia en test, 61 duplicados dentro del mismo split, 2 con etiquetas contradictorias) y 5 imágenes borrosas en train (varianza del Laplaciano < 30). Ninguna por exposición.
- **Splits finales** (`data/splits.csv`): train 7810, valid 693, test 696 (70 razas; valid y test con ~10 por raza).
- **Conjunto independiente** (141 imágenes, nunca vistas en entrenamiento): 119 limpias de 21 razas (Oxford-IIIT Pet, CC BY-SA 4.0, y Stanford Dogs, uso académico), 12 escenas con 2 a 4 perros y 10 gatos. Se eliminaron 7 que coincidían con imágenes de Kaggle. Fuente y licencia por imagen en `fuentes.csv`.
- **Preprocesamiento:** el mismo en la app y en el entrenamiento (`src/lib/preprocessing.py`).
- **Augmentation (solo train, con semilla):** flip horizontal, rotación ±15°, blur, brillo/contraste y ruido gaussiano.

## Modelos y entrenamiento

Detalle completo en `2-train.ipynb`. Entrenamiento en Google Colab (GPU T4).

- **Baseline de embeddings (Etapa 1):** ResNet18 preentrenada en ImageNet, sin entrenar.
- **Modelo A (elegido para clasificar):** ResNet18 ImageNet con cabeza propia (Dropout + Linear). Se compararon dos estrategias: `head` (backbone congelado, lr 1e-3) y **`layer4`** (último bloque descongelado, lr 1e-4), que ganó por accuracy de valid (96.1% contra 94.7%).
- **Modelo B:** CNN propia de 5 bloques convolucionales (32→256 canales), GAP y embedding de 512 dimensiones (1.15 M parámetros), entrenada desde cero. Se probó dropout 0.3 y 0.5; quedó 0.3.
- **Hiperparámetros comunes:** AdamW (weight decay 1e-4), batch 32, scheduler coseno, CrossEntropy con label smoothing 0.1 y early stopping por accuracy de valid (paciencia 5 para A, 8 para B).
- **Detector:** YOLOv8n (COCO, clase 16), `YOLO_CONF_THRESHOLD=0.25`.
- **Umbral de similitud:** 0.82, calibrado con `valid` (percentil 1 del score top-1).
- **Checkpoints:** https://drive.google.com/drive/folders/1WnbeiZCLiiL_dbsBIPEEw1tqZpgT149K?usp=sharing

## Resultados

**Etapa 1 — Búsqueda por similitud** (galería = train, cosine, Top-10):

| Embeddings | NDCG@10 test | NDCG@10 independiente | Accuracy test | Accuracy independiente |
|---|---|---|---|---|
| baseline (ResNet18 ImageNet) | 0.874 | **0.773** | 0.937 | **0.840** |
| Modelo A | **0.942** | 0.767 | **0.961** | 0.798 |
| Modelo B | 0.684 | 0.317 | 0.772 | 0.403 |

**Etapa 2 — Clasificación supervisada** (test, promedio macro):

| Métrica | Modelo A (ResNet18 `layer4`) | Modelo B (CNN propia) |
|---|---|---|
| Accuracy | **0.958** | 0.787 |
| Precision / Recall / F1 | **0.963 / 0.958 / 0.958** | 0.801 / 0.787 / 0.784 |
| Specificity | 0.999 | 0.997 |
| Accuracy (independiente) | **0.773** | 0.370 |
| Parámetros / tamaño | 11.2 M / 45 MB | 1.15 M / 4.6 MB |

**Conclusiones:**

- **Clasificación:** el Modelo A es el mejor en todo; el transfer learning desde ImageNet supera ampliamente a la CNN entrenada desde cero, que es 10 veces más chica pero pierde 17 puntos en test y 40 fuera de dominio.
- **Búsqueda:** los embeddings del Modelo A son mejores dentro del dominio del dataset, pero en fotos reales (conjunto independiente) el baseline es igual o mejor. Como la búsqueda se usa con fotos propias, la app usa `baseline`.
- **Cambio de dominio:** todos los modelos caen en el conjunto independiente (Modelo A: 97.1% → 77.3% con las mismas razas). Test sobreestima el rendimiento real.
- **Errores principales:** razas casi idénticas (Bulldog / Boston Terrier / French Bulldog, German Sheperd / Malinois, Labradoodle / Cockapoo), etiquetas dudosas en la clase Bulldog, y perros lejanos o de cuerpo entero en fotos reales, donde el modelo decide por color y silueta.

## Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| "American  Spaniel" con doble espacio en `valid` aparecía como una raza 71 | Normalización de espacios en las etiquetas (`load_split` y notebook 1) |
| Imágenes duplicadas entre splits (fuga de datos), aunque el autor del dataset indica que no hay | Detección con dHash y filtrado de 147 imágenes; splits guardados en `data/splits.csv` |
| YOLO no detectaba perros en el 45% de las imágenes de 224 px del dataset al usar `imgsz=640` | En el EDA se contó con `imgsz=320` (16% sin detección); en la app llegan fotos reales y se usa el tamaño por defecto |
| `BatchNorm1d` falla con un último batch de una sola imagen | `drop_last=True` en el DataLoader de train |
| El augmentation no era reproducible entre corridas | Semilla en `A.Compose` |
| Entrenar en CPU era demasiado lento | Entrenamiento en Colab con GPU; el checkpoint se guarda en CPU para que la app lo cargue sin GPU |
| Las BatchNorm del backbone congelado actualizaban sus estadísticas | Se dejan en modo eval durante el entrenamiento |
| La base vectorial guarda un solo modelo de embeddings: buscar con otro modelo da resultados sin sentido | Al cambiar `EMBEDDING_MODEL` se reindexa con `build_index.py` |
| El fine-tuning empeora la búsqueda en fotos reales | Se eligió `baseline` para la búsqueda y el Modelo A solo para clasificar |
| Un gato negro es detectado como perro por YOLO (0.60) en cualquier umbral | Limitación del detector preentrenado, documentada |

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Los tres modelos de este trabajo generan 512 dimensiones.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- `build_index.py` indexa la carpeta `train/` completa del dataset (7946 imágenes). Las notebooks usan como galería el train filtrado de `splits.csv` (7810); la diferencia son los duplicados y las imágenes borrosas descartadas.
- Las rutas de la galería se guardan relativas a `DATA_PATH`: una base indexada desde el entorno local también sirve para el backend en Docker.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`).
- Los pesos preentrenados (torchvision y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde `feature/tp2-razas-perros` hacia `main` del fork, sin mergear.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- El sistema funciona con:

```bash
docker compose build
docker compose up
```
