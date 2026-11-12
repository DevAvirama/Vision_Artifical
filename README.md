# PorciTech Vision — Estimación Biométrica de Peso Sin Contacto

Worker contenerizado de visión por computador para el ecosistema **PorciTech**. Procesa flujos de video en tiempo real (`RTSP`) para estimar el peso vivo de ejemplares porcinos sin estrés ni contacto físico, combinando segmentación de instancias con **YOLOv8**, calibración métrica mediante marcadores fiduciarios **ArUco** y regresión tabular con **XGBoost**.

- **Caso de estudio técnico:** [santiagoavirama.dev/proyectos/porcitech](https://santiagoavirama.dev/proyectos/porcitech)
- **Integración Backend:** [`DevAvirama/api_porcitech`](https://github.com/DevAvirama/api_porcitech)

---

## Pipeline de Inferencia Biométrica

```mermaid
flowchart LR
    RTSP[Flujo RTSP Cámara IP] --> CV[OpenCV Frame Capture & Filter]
    CV --> ARUCO[Detección Marcador ArUco<br/>Calibración cm/px]
    CV --> YOLO[YOLOv8 Segmentation<br/>Máscara Dorsal]
    ARUCO --> FEAT[Extracción de Rasgos Métricos<br/>Área, Longitud, Ancho]
    YOLO --> FEAT
    FEAT --> XGB[Modelo Regresor XGBoost<br/>Estimación en kg]
    XGB -->|POST /api/v1/pesajes| API[api_porcitech]
```

### Fases del Algoritmo

1. **Ingesta y Filtrado Óptico (`OpenCV`):** Captura resiliente de cuadros desde cámaras RTSP con descarte automático de fotogramas desenfocados o con oclusión parcial.
2. **Calibración Métrica (`ArUco`):** Identificación de marcadores fiduciarios de dimensiones físicas conocidas en el corral para calcular la relación real $\text{cm}/\text{px}$ y corregir distorsiones de perspectiva.
3. **Segmentación Morfométrica (`YOLOv8`):** Aislamiento de la silueta dorsal del animal para extraer área superficial proyectada, longitud dorsal y perímetro.
4. **Inferencia de Peso (`XGBoost`):** Predicción del peso en kilogramos a partir del vector de características calibradas y envío autenticado hacia `api_porcitech`.

---

## Variables de Entorno

| Variable               | Descripción                                         | Ejemplo                                    |
| :--------------------- | :-------------------------------------------------- | :----------------------------------------- |
| `RTSP_STREAM_URL`      | URI del flujo de video de la cámara IP              | `rtsp://user:pass@192.168.1.50:554/stream` |
| `API_CORE_URL`         | Endpoint base de `api_porcitech`                    | `http://api_porcitech:8000/api/v1`         |
| `WORKER_API_TOKEN`     | Token de autenticación del servicio                 | `eyJhbGciOi...`                            |
| `ARUCO_MARKER_SIZE_CM` | Lado físico real del marcador ArUco instalado       | `15.0`                                     |
| `CONFIDENCE_THRESHOLD` | Umbral mínimo de confianza para segmentación YOLOv8 | `0.80`                                     |

---

## Ejecución en Contenedor

```bash
git clone [https://github.com/DevAvirama/Vision_Artificial.git](https://github.com/DevAvirama/Vision_Artificial.git)
cd Vision_Artificial
cp .env.example .env
docker build -t porcitech-vision .
docker run -d --name porcitech-vision --restart unless-stopped --env-file .env porcitech-vision
```
