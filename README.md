# 👁️ Visión Artificial - Detección y Segmentación con YOLOv8

Proyecto de visión por computadora enfocado en inferencia y segmentación de instancias en tiempo real utilizando modelos **YOLOv8** (`yolov8n-seg`), estructurado con empaquetado moderno en Python y gestionado mediante el administrador de paquetes de alto rendimiento **`uv`**.

---

## 🏗️ Estructura del Proyecto

```text
Vision_Artificial/
├── .python-version        # Versión de Python fijada para el entorno
├── pyproject.toml         # Configuración del paquete y dependencias
├── uv.lock                # Bloqueo determinista de dependencias
├── yolov8n-seg.pt         # Pesos preentrenados de YOLOv8 (Segmentación Nano)
├── visionArtificial.py    # Script principal de ejecución e inferencia
└── src/
    └── vision_artificial/
        └── __init__.py    # Módulo raíz y punto de entrada de la aplicación
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Rol en el Proyecto |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.12+ 🐍 | Lenguaje base para procesamiento y pipelines de visión. |
| **Framework de IA** | Ultralytics / YOLOv8 🧠 | Modelo de visión artificial para segmentación de instancias. |
| **Procesamiento de Imagen** | OpenCV (`cv2`) 📷 | Captura de video/cámara y renderizado de máscaras. |
| **Gestor de Paquetes** | `uv` ⚡ | Creación de entornos virtuales ultrarrápidos y resolución de dependencias. |
| **Backend de Tensores** | PyTorch 🔥 | Motor de inferencia y aceleración por GPU/CPU. |

---

## ✨ Características Principales

* 🎯 **Segmentación de Instancias:** Detección de objetos y delimitación precisa de contornos/máscaras a nivel de píxel.
* ⚡ **Modelo Ultraliviano:** Uso de `yolov8n-seg.pt` optimizado para alta tasa de cuadros por segundo (FPS) en CPUs y GPUs.
* 📦 **Gestión Moderna con `uv`:** Entorno virtual reproducible y tiempos de instalación en milisegundos.
* 🔄 **Soporte Multi-fuente:** Procesamiento directo desde cámara web en vivo, archivos de video o imágenes estáticas.

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone [https://github.com/tu-usuario/Vision_Artificial.git](https://github.com/tu-usuario/Vision_Artificial.git)
cd Vision_Artificial
```

### 2. Crear y activar el entorno virtual con `uv`

* **En terminal `fish` (Garuda Linux):**
  ```fish
  uv venv
  source .venv/bin/activate.fish
  ```

* **En terminal `bash` / `zsh`:**
  ```bash
  uv venv
  source .venv/bin/activate
  ```

### 3. Instalar dependencias

Sincroniza el entorno directamente desde el archivo `uv.lock` o `pyproject.toml`:
```bash
uv sync
```

*(Alternativa manual con `uv pip`):*
```bash
uv pip install ultralytics opencv-python torch torchvision
```

---

## 🎮 Ejecución y Pruebas

### Inferencia básica con el script principal
```bash
uv run python visionArtificial.py
```

### Inferencia directa desde el módulo
```bash
uv run python -m vision_artificial
```

### Ejemplo de uso en código (`visionArtificial.py`)
```python
from ultralytics import YOLO
import cv2

# Cargar el modelo de segmentación
model = YOLO("yolov8n-seg.pt")

# Ejecutar inferencia en vivo con la cámara web (índice 0)
results = model.predict(source="0", show=True, conf=0.5)
```

---

## ⚙️ Comandos de Mantenimiento y Control

```bash
# Agregar una nueva librería al proyecto
uv add numpy

# Actualizar todas las dependencias
uv lock --upgrade

# Ejecutar pruebas o scripts dentro del entorno
uv run python visionArtificial.py
```
