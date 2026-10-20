import os
import time
import base64
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from ultralytics import YOLO

# Forzar transporte TCP y timeout de 5s en FFMPEG para streams RTSP de Dahua
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

DAHUA_DEFAULT_RTSP = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# ============================================================================
# 1. CONSTANTES DE CALIBRACIÓN ALOMÉTRICA Y ESCALA CENITAL
# ============================================================================
DEFAULT_REF_WIDTH = 1280
DEFAULT_PIXELS_PER_CM_AT_1280 = 15.2  # Calibrado para cámara cenital a ~2.20m de altura
COEF_ALOMETRICO_A = 0.0245
EXPONENTE_ALOMETRICO_B = 1.052
FACTOR_OCUPACION_DORSAL = 0.76
MIN_AREA_PIXELES = 3000.0


# ============================================================================
# 2. ESQUEMAS PYDANTIC V2 (TIPADO ESTRICTO)
# ============================================================================
class BoundingBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    ancho_px: int
    alto_px: int
    resolucion: str
    pixels_per_cm_aplicado: float


class DeteccionIndividualSchema(BaseModel):
    confianza: float = Field(..., ge=0.0, le=1.0)
    peso_estimado_kg: float = Field(..., gt=0.0)
    area_pixeles: float
    area_cm2: float
    largo_cm: float
    ancho_cm: float
    bbox: BoundingBoxSchema
    bounding_box: Dict[str, int]  # Retrocompatibilidad con clientes previos


class InferenciaVisionResponse(BaseModel):
    status: str = "ok"
    fecha_registro: datetime
    total_detectados: int
    # Métricas del ejemplar principal (listas para insertar en registro_pesos de TimescaleDB)
    peso_estimado_kg: float
    confianza: float
    area_cm2: float
    largo_cm: float
    ancho_cm: float
    bbox: Dict[str, Any]
    imagen_anotada_base64: str
    detecciones: List[DeteccionIndividualSchema]

    model_config = ConfigDict(from_attributes=True)


class CapturaRTSPRequest(BaseModel):
    rtsp_url: Optional[str] = Field(
        None,
        description="URL RTSP de la cámara V380 Pro (ej. rtsp://admin:123456@192.168.32.200:554/live/ch00_1)"
    )
    pixels_per_cm: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Calibración opcional en px/cm sobre base 1280px"
    )
    conf_threshold: float = Field(0.45, ge=0.1, le=0.99)
    validar_bordes: bool = Field(
        False, description="Si es True, rechaza capturas donde el cerdo toque el borde de la imagen"
    )


# ============================================================================
# 3. MANEJADOR DE CONTEXTO LIFESPAN (INSTANCIACIÓN ÚNICA DEL MODELO YOLO)
# ============================================================================
MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB máximo por imagen


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("YOLO_MODEL_PATH", "best.pt")
    print(f"[PorciTech Vision] Cargando modelo YOLO en memoria única desde: {model_path}...")
    yolo_model = YOLO(model_path)

    # Warmup con tensor vacío para inicializar memoria antes de la primera petición HTTP
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo_model.predict(source=dummy_frame, verbose=False)

    app.state.model = yolo_model
    app.state.yolo_lock = asyncio.Lock()
    print("[PorciTech Vision] Modelo cargado y listo para inferencia de baja latencia.")
    yield
    app.state.model = None


app = FastAPI(
    title="Microservicio de Visión PorciTech",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 4. FUNCIONES SÍNCRONAS CPU-BOUND (EJECUTADAS EN THREADPOOL)
# ============================================================================
def decodificar_imagen_ram_sync(contents: bytes) -> np.ndarray:
    """Decodifica los bytes recibidos por UploadFile directamente en RAM."""
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("No se pudo decodificar la imagen enviada.")
    return frame


def capturar_frame_v380_rtsp_sync(
    rtsp_url: str,
    warmup_grabs: int = 12,
    timeout_sec: float = 6.0
) -> np.ndarray:
    """
    Se conecta por TCP al stream RTSP de la cámara Wi-Fi V380 Pro,
    drena el buffer de frames acumulados y devuelve el fotograma en vivo.
    """
    inicio = time.monotonic()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise ConnectionError(
            f"No fue posible conectar con la cámara V380 Pro en '{rtsp_url}'. "
            "Verifique que RTSP esté activo en la cámara y que la IP sea alcanzable."
        )

    try:
        for _ in range(warmup_grabs):
            if (time.monotonic() - inicio) > timeout_sec:
                break
            cap.grab()

        exito, frame = cap.retrieve()
        if not exito or frame is None or frame.size == 0:
            exito, frame = cap.read()
            if not exito or frame is None or frame.size == 0:
                raise ValueError("La cámara V380 Pro entregó un fotograma vacío o corrupto.")
        return frame
    finally:
        cap.release()


def procesar_frame_morfometria_sync(
    model: YOLO,
    frame: np.ndarray,
    pixels_per_cm_override: Optional[float] = None,
    conf_threshold: float = 0.45,
    validar_bordes: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta inferencia YOLO + OpenCV en el threadpool sin bloquear el Event Loop.
    Calcula área dorsal real, largo_cm, ancho_cm y peso alométrico normalizado.
    """
    alto_orig, ancho_orig = frame.shape[:2]

    # 1. Normalización dinámica de escala según la resolución de la imagen
    factor_resolucion = ancho_orig / float(DEFAULT_REF_WIDTH)
    base_ppc = (
        pixels_per_cm_override
        if (pixels_per_cm_override and pixels_per_cm_override > 0)
        else DEFAULT_PIXELS_PER_CM_AT_1280
    )
    pixels_per_cm_efectivo = base_ppc * factor_resolucion

    # 2. Inferencia con el modelo único de app.state.model
    resultados = model.predict(source=frame, conf=conf_threshold, verbose=False)
    if not resultados or len(resultados[0].boxes) == 0:
        raise ValueError("No se detectó ningún ejemplar porcino en la imagen.")

    res = resultados[0]
    boxes = res.boxes
    tiene_mascaras = res.masks is not None and len(res.masks.xy) > 0

    img_anotada = frame.copy()
    detecciones: List[DeteccionIndividualSchema] = []

    for idx in range(len(boxes)):
        conf = float(boxes.conf[idx].cpu().numpy())
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].cpu().numpy()]

        # Validación opcional de bordes para cámara cenital fija
        if validar_bordes:
            margen_px = 6.0
            if (
                x1 <= margen_px
                or y1 <= margen_px
                or x2 >= (ancho_orig - margen_px)
                or y2 >= (alto_orig - margen_px)
            ):
                continue

        # 3. Extracción de geometría dorsal (Máscara poligonal YOLO-seg o Elipse YOLO-detect)
        if tiene_mascaras and idx < len(res.masks.xy) and len(res.masks.xy[idx]) >= 3:
            contorno = np.array(res.masks.xy[idx], dtype=np.int32)
            area_pixeles = float(cv2.contourArea(contorno))
            if area_pixeles < MIN_AREA_PIXELES:
                continue

            # Rectángulo rotado de área mínima para obtener largo y ancho reales sin importar orientación
            (_, _), (dim1_px, dim2_px), _ = cv2.minAreaRect(contorno)
            largo_px = float(max(dim1_px, dim2_px))
            ancho_px = float(min(dim1_px, dim2_px))
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contorno)

            cv2.polylines(img_anotada, [contorno], isClosed=True, color=(16, 185, 129), thickness=3)
        else:
            w_rect = int(max(x2 - x1, 1.0))
            h_rect = int(max(y2 - y1, 1.0))
            x_rect, y_rect = int(x1), int(y1)
            area_bbox = float(w_rect * h_rect)
            if area_bbox < MIN_AREA_PIXELES:
                continue

            largo_px = float(max(w_rect, h_rect))
            ancho_px = float(min(w_rect, h_rect))
            area_pixeles = area_bbox * FACTOR_OCUPACION_DORSAL

        # 4. Conversión a centímetros y cálculo alométrico de peso
        largo_cm = round(largo_px / pixels_per_cm_efectivo, 2)
        ancho_cm = round(ancho_px / pixels_per_cm_efectivo, 2)
        area_cm2 = round(area_pixeles / (pixels_per_cm_efectivo ** 2), 2)
        peso_estimado_kg = round(COEF_ALOMETRICO_A * (area_cm2 ** EXPONENTE_ALOMETRICO_B), 2)

        bbox_obj = BoundingBoxSchema(
            x1=round(x1, 1),
            y1=round(y1, 1),
            x2=round(x2, 1),
            y2=round(y2, 1),
            ancho_px=w_rect,
            alto_px=h_rect,
            resolucion=f"{ancho_orig}x{alto_orig}",
            pixels_per_cm_aplicado=round(pixels_per_cm_efectivo, 3)
        )

        detecciones.append(
            DeteccionIndividualSchema(
                confianza=round(conf, 4),
                peso_estimado_kg=peso_estimado_kg,
                area_pixeles=round(area_pixeles, 1),
                area_cm2=area_cm2,
                largo_cm=largo_cm,
                ancho_cm=ancho_cm,
                bbox=bbox_obj,
                bounding_box={"x": x_rect, "y": y_rect, "ancho": w_rect, "alto": h_rect}
            )
        )

        # Dibujar anotación visual sobre la imagen de evidencia
        cv2.rectangle(img_anotada, (int(x1), int(y1)), (int(x2), int(y2)), (16, 185, 129), 3)
        etiqueta = f"Porcino: {conf*100:.1f}% | {peso_estimado_kg} kg"
        cv2.putText(
            img_anotada,
            etiqueta,
            (int(x1), max(int(y1) - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (16, 185, 129),
            2
        )

    if not detecciones:
        raise ValueError(
            "No se detectó ningún ejemplar porcino válido (o el ejemplar se encuentra cortado en el borde)."
        )

    # Ordenar detecciones priorizando la mayor confianza
    detecciones.sort(key=lambda d: d.confianza, reverse=True)
    principal = detecciones[0]

    # Codificar imagen anotada en JPEG Base64 en memoria
    _, buffer_jpg = cv2.imencode(".jpg", img_anotada, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    imagen_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_jpg).decode('utf-8')}"

    return {
        "fecha_registro": datetime.now(timezone.utc),
        "total_detectados": len(detecciones),
        "peso_estimado_kg": principal.peso_estimado_kg,
        "confianza": principal.confianza,
        "area_cm2": principal.area_cm2,
        "largo_cm": principal.largo_cm,
        "ancho_cm": principal.ancho_cm,
        "bbox": principal.bbox.model_dump(),
        "imagen_anotada_base64": imagen_base64,
        "detecciones": detecciones
    }


# ============================================================================
# 5. ENDPOINTS DE LA API DE VISIÓN
# ============================================================================
@app.get("/health")
async def health_check(request: Request):
    """Verifica el estado del microservicio y que el modelo YOLO esté cargado en RAM."""
    modelo_listo = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok" if modelo_listo else "degraded",
        "service": "vision-artificial",
        "model_loaded": modelo_listo
    }


@app.post("/estimar-peso", response_model=InferenciaVisionResponse)
async def estimar_peso(
    request: Request,
    file: UploadFile = File(...),
    pixels_per_cm: Optional[float] = Form(None),
    conf_threshold: float = Form(0.45),
    validar_bordes: bool = Form(False)
):
    """
    Recibe una imagen vía multipart/form-data (Next.js o Flutter),
    la decodifica en memoria RAM y ejecuta la inferencia en el threadpool.
    """
    contents = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")
    if len(contents) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el tamaño máximo permitido de 15 MB.",
        )

    yolo_lock: Optional[asyncio.Lock] = getattr(request.app.state, "yolo_lock", None)

    try:
        frame = await asyncio.to_thread(decodificar_imagen_ram_sync, contents)
        if yolo_lock is not None:
            async with yolo_lock:
                resultado = await asyncio.to_thread(
                    procesar_frame_morfometria_sync,
                    model=request.app.state.model,
                    frame=frame,
                    pixels_per_cm_override=pixels_per_cm,
                    conf_threshold=conf_threshold,
                    validar_bordes=validar_bordes
                )
        else:
            resultado = await asyncio.to_thread(
                procesar_frame_morfometria_sync,
                model=request.app.state.model,
                frame=frame,
                pixels_per_cm_override=pixels_per_cm,
                conf_threshold=conf_threshold,
                validar_bordes=validar_bordes
            )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/estimar-peso-rtsp", response_model=InferenciaVisionResponse)
async def estimar_peso_desde_camara_rtsp(
    payload: CapturaRTSPRequest,
    request: Request
):
    """
    Se conecta directamente al stream RTSP de la cámara Wi-Fi V380 Pro,
    captura el fotograma en vivo sin lag y calcula el peso corporal estimado.
    """
    rtsp_url = (payload.rtsp_url or os.getenv("V380_RTSP_URL") or "").strip()
    if not rtsp_url:
        raise HTTPException(
            status_code=400,
            detail="Debe enviar 'rtsp_url' en el JSON o definir V380_RTSP_URL en las variables de entorno."
        )
    if not rtsp_url.lower().startswith(("rtsp://", "rtsps://")):
        raise HTTPException(
            status_code=400,
            detail="La URL de la cámara debe utilizar el protocolo rtsp:// o rtsps://."
        )

    try:
        frame = await asyncio.to_thread(capturar_frame_v380_rtsp_sync, rtsp_url)
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=504, detail=str(e))

    yolo_lock: Optional[asyncio.Lock] = getattr(request.app.state, "yolo_lock", None)

    try:
        if yolo_lock is not None:
            async with yolo_lock:
                resultado = await asyncio.to_thread(
                    procesar_frame_morfometria_sync,
                    model=request.app.state.model,
                    frame=frame,
                    pixels_per_cm_override=payload.pixels_per_cm,
                    conf_threshold=payload.conf_threshold,
                    validar_bordes=payload.validar_bordes
                )
        else:
            resultado = await asyncio.to_thread(
                procesar_frame_morfometria_sync,
                model=request.app.state.model,
                frame=frame,
                pixels_per_cm_override=payload.pixels_per_cm,
                conf_threshold=payload.conf_threshold,
                validar_bordes=payload.validar_bordes
            )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))