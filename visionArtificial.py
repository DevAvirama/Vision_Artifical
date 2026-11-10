import os
import json
import time
import base64
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from ultralytics import YOLO

# Forzar transporte TCP y timeout de 5s en FFMPEG para streams RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

DAHUA_DEFAULT_RTSP = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
ARCHIVO_CALIBRACION = Path("calibracion_camara.json")

# ============================================================================
# 1. CONSTANTES DE CALIBRACIÓN ALOMÉTRICA Y ESCALA CENITAL
# ============================================================================
DEFAULT_REF_WIDTH = 1280
DEFAULT_PIXELS_PER_CM_AT_1280 = 15.2  # Fallback de fábrica para cámara cenital a ~2.20m de altura
COEF_ALOMETRICO_A = 0.0245
EXPONENTE_ALOMETRICO_B = 1.052
FACTOR_OCUPACION_DORSAL = 0.76
MIN_AREA_PIXELES = 3000.0

# Configuración de marcador ArUco por defecto
ARUCO_DICCIONARIO_DEFAULT = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE_CM = 10.0


# ============================================================================
# 2. ESQUEMAS PYDANTIC V2
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
    ancho_toracico_cm: float
    ratio_aspecto: float
    momentos_hu: List[float] = Field(default_factory=list, description="7 Momentos invariantes de Hu en escala logarítmica")
    vector_caracteristicas_xgboost: List[float] = Field(default_factory=list, description="Vector listo para inferencia/entrenamiento con XGBoost")
    bbox: BoundingBoxSchema
    bounding_box: Dict[str, int]


class ConceptoVeterinarioSchema(BaseModel):
    condicion_corporal_bcs: Optional[int] = Field(None, ge=1, le=5, description="Puntuación de Condición Corporal (BCS) de 1 (emaciado) a 5 (obeso)")
    clasificacion_bcs: Optional[str] = Field(None, description="Clasificación cualitativa: Emaciado, Delgado, Ideal, Sobrepeso, Obeso")
    alerta_canibalismo_lesiones: bool = Field(False, description="True si se detectan mordeduras de cola (tail-biting), orejas o laceraciones")
    descripcion_lesiones: Optional[str] = Field(None, description="Detalle clínico de las lesiones detectadas")
    evaluacion_clinica_general: str = Field(..., description="Concepto zootécnico general sobre el desarrollo corporal")
    recomendaciones_manejo: str = Field(..., description="Recomendaciones nutricionales o sanitarias preventivas")


class InferenciaVisionResponse(BaseModel):
    status: str = "ok"
    fecha_registro: datetime
    total_detectados: int
    metodo_calibracion: str  # "aruco_guardado", "aruco_en_vivo", "manual_parametro", "fabrica_cenital"
    algoritmo_peso: str      # "alomorfico_huxley" o "xgboost"
    pixels_per_cm_utilizado: float
    peso_estimado_kg: float
    confianza: float
    area_cm2: float
    largo_cm: float
    ancho_cm: float
    ancho_toracico_cm: float
    bbox: Dict[str, Any]
    imagen_anotada_base64: str
    detecciones: List[DeteccionIndividualSchema]
    concepto_veterinario: Optional[ConceptoVeterinarioSchema] = Field(None, description="Evaluación preliminar de salud y bienestar emitida por Gemini Multimodal")

    model_config = ConfigDict(from_attributes=True)


class CalibracionRespuesta(BaseModel):
    status: str = "ok"
    mensaje: str
    pixels_per_cm_base1280: float
    tamano_marcador_cm: float
    marcador_id: int
    fecha_calibracion: str
    resolucion_captura: str
    imagen_anotada_base64: str


class CapturaRTSPRequest(BaseModel):
    rtsp_url: Optional[str] = Field(
        None,
        description="URL RTSP de la cámara (ej. rtsp://admin:123456@192.168.1.108:554/live/ch00_1)"
    )
    pixels_per_cm: Optional[float] = Field(
        None, gt=1.0, le=100.0, description="Override manual opcional de calibración"
    )
    conf_threshold: float = Field(0.45, ge=0.1, le=0.99)
    validar_bordes: bool = Field(
        False, description="Si es True, rechaza capturas donde el cerdo toque el borde de la imagen"
    )
    evaluar_veterinaria: bool = Field(
        False, description="Si es True, activa el análisis clínico multimodal con la API de Gemini"
    )
    gemini_api_key: Optional[str] = Field(
        None, description="API Key opcional de Google Gemini (o toma la variable GEMINI_API_KEY)"
    )


class CalibrarRTSPRequest(BaseModel):
    rtsp_url: str = Field(..., description="URL RTSP de la cámara apuntando al corral con el marcador en el suelo")
    tamano_marcador_cm: float = Field(10.0, gt=1.0, le=100.0, description="Tamaño real del lado del marcador en cm")
    diccionario_aruco: int = Field(ARUCO_DICCIONARIO_DEFAULT, description="ID del diccionario ArUco de OpenCV")


# ============================================================================
# 3. GESTIÓN DE CALIBRACIÓN PERSISTENTE (JSON LOCAL)
# ============================================================================
def cargar_calibracion_persistente() -> Optional[Dict[str, Any]]:
    """Carga los parámetros guardados de calibración métrica desde calibracion_camara.json."""
    if ARCHIVO_CALIBRACION.exists():
        try:
            with open(ARCHIVO_CALIBRACION, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[PorciTech Vision] Advertencia al leer {ARCHIVO_CALIBRACION}: {e}")
    return None


def guardar_calibracion_persistente(
    pixels_per_cm_1280: float,
    tamano_marcador_cm: float,
    marcador_id: int,
    resolucion: str
) -> Dict[str, Any]:
    """Persiste en disco la calibración métrica obtenida con el marcador ArUco."""
    datos = {
        "pixels_per_cm_base1280": round(pixels_per_cm_1280, 4),
        "tamano_marcador_cm": tamano_marcador_cm,
        "marcador_id": marcador_id,
        "resolucion_calibracion": resolucion,
        "ancho_referencia": DEFAULT_REF_WIDTH,
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "estado": "calibrado"
    }
    with open(ARCHIVO_CALIBRACION, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    return datos


# ============================================================================
# 4. FUNCIONES DE DETECCIÓN ARUCO Y MORFOMETRÍA AVANZADA
# ============================================================================
def detectar_marcador_aruco(
    frame: np.ndarray,
    tamano_marcador_cm: float = DEFAULT_MARKER_SIZE_CM,
    diccionario_id: int = ARUCO_DICCIONARIO_DEFAULT
) -> Optional[Dict[str, Any]]:
    """
    Detecta el marcador fiducial ArUco en el frame y calcula la escala exacta píxeles/cm.
    Dibuja el contorno cian del marcador sobre el frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(diccionario_id)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    esquinas, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(esquinas) > 0:
        # Tomar el primer marcador detectado
        pts = esquinas[0][0]  # Array de 4 puntos (x, y)
        marker_id = int(ids.flatten()[0])

        lado_px_1 = float(np.linalg.norm(pts[0] - pts[1]))
        lado_px_2 = float(np.linalg.norm(pts[1] - pts[2]))
        lado_px_3 = float(np.linalg.norm(pts[2] - pts[3]))
        lado_px_4 = float(np.linalg.norm(pts[3] - pts[0]))
        lado_px_promedio = (lado_px_1 + lado_px_2 + lado_px_3 + lado_px_4) / 4.0

        if lado_px_promedio > 10.0 and tamano_marcador_cm > 0:
            ppc_medido = float(lado_px_promedio / tamano_marcador_cm)
            alto, ancho = frame.shape[:2]
            # Normalizar a base 1280px para independencia de resolución
            ppc_base1280 = ppc_medido / (ancho / float(DEFAULT_REF_WIDTH))

            # Dibujar anotación visual del marcador ArUco en color cian (255, 255, 0 en BGR)
            cv2.aruco.drawDetectedMarkers(frame, esquinas, ids, borderColor=(255, 255, 0))
            cv2.putText(
                frame,
                f"ArUco ID {marker_id}: {ppc_medido:.2f} px/cm",
                (int(pts[0][0]), max(int(pts[0][1]) - 15, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            return {
                "pixels_per_cm_medido": ppc_medido,
                "pixels_per_cm_base1280": ppc_base1280,
                "marcador_id": marker_id,
                "lado_px_promedio": lado_px_promedio,
                "esquinas": pts.tolist()
            }
    return None


def calcular_momentos_hu(contorno: np.ndarray) -> List[float]:
    """Calcula los 7 Momentos Invariantes de Hu en escala logarítmica simétrica."""
    m = cv2.moments(contorno)
    hu = cv2.HuMoments(m).flatten()
    hu_log = []
    for h in hu:
        if abs(h) > 1e-15:
            hu_log.append(round(-float(np.sign(h) * np.log10(abs(h))), 4))
        else:
            hu_log.append(0.0)
    return hu_log


def calcular_ancho_toracico_px(contorno: np.ndarray, largo_px: float, ancho_px: float) -> float:
    """
    Calcula el ancho torácico dorsal mediante corte transversal en la franja
    anterior-media de la caja torácica (entre el 30% y 45% de la longitud del cuerpo).
    """
    try:
        rect = cv2.minAreaRect(contorno)
        (cx, cy), (w, h), angulo = rect

        # Orientar la dimensión mayor a lo largo del eje horizontal X
        if w < h:
            angulo += 90.0

        mat_rot = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        pts = contorno.reshape(-1, 2)
        pts_rot = cv2.transform(np.array([pts]), mat_rot)[0]

        x_min, x_max = np.min(pts_rot[:, 0]), np.max(pts_rot[:, 0])
        longitud_real = x_max - x_min
        if longitud_real <= 0:
            return float(ancho_px)

        # Franja torácica: 30% a 45% desde el extremo craneal
        x_inicio = x_min + 0.30 * longitud_real
        x_fin = x_min + 0.45 * longitud_real
        pts_torax = pts_rot[(pts_rot[:, 0] >= x_inicio) & (pts_rot[:, 0] <= x_fin)]

        if len(pts_torax) >= 2:
            ancho_torax_px = float(np.max(pts_torax[:, 1]) - np.min(pts_torax[:, 1]))
            if 0.4 * ancho_px <= ancho_torax_px <= 1.4 * ancho_px:
                return round(ancho_torax_px, 2)

        # Fallback anatómico porcino estándar: ~92% del ancho dorsal máximo
        return round(float(ancho_px) * 0.92, 2)
    except Exception:
        return round(float(ancho_px) * 0.92, 2)


# ============================================================================
# 5. LIFESPAN (INICIALIZACIÓN DE MODELOS)
# ============================================================================
MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Cargar modelo YOLO
    model_path = os.getenv("YOLO_MODEL_PATH", "best.pt")
    if not os.path.exists(model_path) and os.path.exists("yolov8n-seg.pt"):
        model_path = "yolov8n-seg.pt"

    print(f"[PorciTech Vision] Cargando modelo YOLO desde: {model_path}...")
    yolo_model = YOLO(model_path)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo_model.predict(source=dummy_frame, verbose=False)
    app.state.model = yolo_model
    app.state.yolo_lock = asyncio.Lock()

    # 2. Verificar modelo XGBoost opcional
    xgb_path = os.getenv("XGBOOST_MODEL_PATH", "modelo_peso_xgboost.json")
    app.state.xgb_model = None
    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb
            app.state.xgb_model = xgb.Booster()
            app.state.xgb_model.load_model(xgb_path)
            print(f"[PorciTech Vision] Modelo XGBoost cargado exitosamente desde: {xgb_path}")
        except Exception as e:
            print(f"[PorciTech Vision] No se pudo inicializar XGBoost ({e}). Usando fórmula alométrica.")
    else:
        print("[PorciTech Vision] No hay modelo XGBoost local. Utilizando fórmula alométrica calibrada de Huxley.")

    # 3. Cargar calibración persistente
    calib = cargar_calibracion_persistente()
    if calib:
        print(f"[PorciTech Vision] Calibración ArUco cargada de disco: {calib.get('pixels_per_cm_base1280')} px/cm (base 1280)")
    else:
        print(f"[PorciTech Vision] Sin calibración previa. Usando escala de fábrica ({DEFAULT_PIXELS_PER_CM_AT_1280} px/cm).")

    yield

    app.state.model = None
    app.state.xgb_model = None


app = FastAPI(
    title="Microservicio de Visión PorciTech con Calibración ArUco",
    version="2.1.0",
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
# 6. PIPELINE MORFOMÉTRICO Y ESTIMACIÓN DE PESO
# ============================================================================
def decodificar_imagen_ram_sync(contents: bytes) -> np.ndarray:
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
    inicio = time.monotonic()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise ConnectionError(
            f"No fue posible conectar con la cámara en '{rtsp_url}'. "
            "Verifique que RTSP esté activo y que la IP sea alcanzable."
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
                raise ValueError("La cámara entregó un fotograma vacío o corrupto.")
        return frame
    finally:
        cap.release()


def procesar_frame_morfometria_sync(
    model: YOLO,
    xgb_model: Optional[Any],
    frame: np.ndarray,
    pixels_per_cm_override: Optional[float] = None,
    conf_threshold: float = 0.45,
    validar_bordes: bool = False
) -> Dict[str, Any]:
    alto_orig, ancho_orig = frame.shape[:2]
    img_anotada = frame.copy()

    # 1. Determinación de la escala métrica (Prioridades: 1. ArUco en vivo, 2. Override manual, 3. Calibración guardada, 4. Fábrica)
    metodo_calibracion = "fabrica_cenital"
    base_ppc = DEFAULT_PIXELS_PER_CM_AT_1280

    # Intento de detección ArUco en el fotograma actual
    resultado_aruco = detectar_marcador_aruco(img_anotada)
    if resultado_aruco is not None:
        base_ppc = resultado_aruco["pixels_per_cm_base1280"]
        metodo_calibracion = "aruco_en_vivo"
    elif pixels_per_cm_override and pixels_per_cm_override > 0:
        base_ppc = pixels_per_cm_override
        metodo_calibracion = "manual_parametro"
    else:
        # Cargar calibración persistente de disco si no hay marcador en la foto
        calib = cargar_calibracion_persistente()
        if calib and "pixels_per_cm_base1280" in calib:
            base_ppc = float(calib["pixels_per_cm_base1280"])
            metodo_calibracion = "aruco_guardado"

    factor_resolucion = ancho_orig / float(DEFAULT_REF_WIDTH)
    pixels_per_cm_efectivo = base_ppc * factor_resolucion

    # 2. Inferencia con YOLO
    resultados = model.predict(source=frame, conf=conf_threshold, verbose=False)
    if not resultados or len(resultados[0].boxes) == 0:
        raise ValueError("No se detectó ningún ejemplar porcino en la imagen.")

    res = resultados[0]
    boxes = res.boxes
    tiene_mascaras = res.masks is not None and len(res.masks.xy) > 0
    detecciones: List[DeteccionIndividualSchema] = []

    for idx in range(len(boxes)):
        conf = float(boxes.conf[idx].cpu().numpy())
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].cpu().numpy()]

        if validar_bordes:
            margen_px = 6.0
            if (
                x1 <= margen_px
                or y1 <= margen_px
                or x2 >= (ancho_orig - margen_px)
                or y2 >= (alto_orig - margen_px)
            ):
                continue

        # 3. Geometría y contornos
        if tiene_mascaras and idx < len(res.masks.xy) and len(res.masks.xy[idx]) >= 3:
            contorno = np.array(res.masks.xy[idx], dtype=np.int32)
            area_pixeles = float(cv2.contourArea(contorno))
            if area_pixeles < MIN_AREA_PIXELES:
                continue

            (_, _), (dim1_px, dim2_px), _ = cv2.minAreaRect(contorno)
            largo_px = float(max(dim1_px, dim2_px))
            ancho_px = float(min(dim1_px, dim2_px))
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contorno)

            # Cálculo de Ancho Torácico y Momentos de Hu
            ancho_torax_px = calcular_ancho_toracico_px(contorno, largo_px, ancho_px)
            momentos_hu = calcular_momentos_hu(contorno)

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
            ancho_torax_px = float(ancho_px * 0.92)
            momentos_hu = [0.0] * 7

        # 4. Conversión métrica
        largo_cm = round(largo_px / pixels_per_cm_efectivo, 2)
        ancho_cm = round(ancho_px / pixels_per_cm_efectivo, 2)
        ancho_toracico_cm = round(ancho_torax_px / pixels_per_cm_efectivo, 2)
        area_cm2 = round(area_pixeles / (pixels_per_cm_efectivo ** 2), 2)
        ratio_aspecto = round(largo_cm / max(ancho_cm, 0.01), 2)

        # Vector listo para XGBoost
        vector_xgboost = [area_cm2, largo_cm, ancho_cm, ancho_toracico_cm, ratio_aspecto] + momentos_hu

        # 5. Inferencia de Peso (XGBoost o Ecuación Alométrica de Huxley)
        algoritmo_peso = "alomorfico_huxley"
        if xgb_model is not None:
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(np.array([vector_xgboost], dtype=np.float32))
                peso_predicho = float(xgb_model.predict(dmatrix)[0])
                peso_estimado_kg = round(peso_predicho, 2)
                algoritmo_peso = "xgboost"
            except Exception as e:
                print(f"[PorciTech Vision] Error en XGBoost predict ({e}), usando fallback Huxley.")
                peso_estimado_kg = round(COEF_ALOMETRICO_A * (area_cm2 ** EXPONENTE_ALOMETRICO_B), 2)
        else:
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
                ancho_toracico_cm=ancho_toracico_cm,
                ratio_aspecto=ratio_aspecto,
                momentos_hu=momentos_hu,
                vector_caracteristicas_xgboost=vector_xgboost,
                bbox=bbox_obj,
                bounding_box={"x": x_rect, "y": y_rect, "ancho": w_rect, "alto": h_rect}
            )
        )

        # Anotación visual
        cv2.rectangle(img_anotada, (int(x1), int(y1)), (int(x2), int(y2)), (16, 185, 129), 3)
        etiqueta = f"Porcino: {conf*100:.1f}% | {peso_estimado_kg} kg ({algoritmo_peso})"
        cv2.putText(
            img_anotada,
            etiqueta,
            (int(x1), max(int(y1) - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (16, 185, 129),
            2
        )

    if not detecciones:
        raise ValueError("No se detectó ningún ejemplar porcino válido.")

    detecciones.sort(key=lambda d: d.confianza, reverse=True)
    principal = detecciones[0]

    _, buffer_jpg = cv2.imencode(".jpg", img_anotada, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    imagen_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_jpg).decode('utf-8')}"

    return {
        "fecha_registro": datetime.now(timezone.utc),
        "total_detectados": len(detecciones),
        "metodo_calibracion": metodo_calibracion,
        "algoritmo_peso": algoritmo_peso,
        "pixels_per_cm_utilizado": round(pixels_per_cm_efectivo, 3),
        "peso_estimado_kg": principal.peso_estimado_kg,
        "confianza": principal.confianza,
        "area_cm2": principal.area_cm2,
        "largo_cm": principal.largo_cm,
        "ancho_cm": principal.ancho_cm,
        "ancho_toracico_cm": principal.ancho_toracico_cm,
        "bbox": principal.bbox.model_dump(),
        "imagen_anotada_base64": imagen_base64,
        "detecciones": detecciones,
        "concepto_veterinario": None
    }


# ============================================================================
# 7. EVALUACIÓN CLÍNICA MULTIMODAL CON GEMINI PRO / FLASH
# ============================================================================
async def evaluar_concepto_veterinario_gemini_async(
    imagen_base64: str,
    peso_kg: float,
    area_cm2: float,
    largo_cm: float,
    ancho_toracico_cm: float,
    api_key_override: Optional[str] = None
) -> Optional[ConceptoVeterinarioSchema]:
    """
    Envía la imagen anotada a la API multimodal de Gemini para emitir un diagnóstico
    veterinario preliminar sobre condición corporal (BCS), bienestar y lesiones.
    """
    api_key = api_key_override or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[PorciTech Gemini] No se proporcionó GEMINI_API_KEY. Omitiendo evaluación clínica.")
        return None

    pure_b64 = imagen_base64
    if "," in pure_b64:
        pure_b64 = pure_b64.split(",", 1)[1]

    prompt = (
        "Actúa como un médico veterinario zootecnista especialista en producción porcina y bienestar animal. "
        "Analiza la siguiente imagen cenital del cerdo evaluado. "
        f"Métricas estimadas por visión artificial: Peso: {peso_kg} kg, Área dorsal: {area_cm2} cm², "
        f"Longitud: {largo_cm} cm, Ancho torácico: {ancho_toracico_cm} cm.\n\n"
        "Debes emitir un concepto veterinario preliminar respondiendo OBLIGATORIAMENTE en formato JSON puro con estas claves:\n"
        "{\n"
        '  "condicion_corporal_bcs": <int del 1 al 5>,\n'
        '  "clasificacion_bcs": "<Emaciado|Delgado|Ideal|Sobrepeso|Obeso>",\n'
        '  "alerta_canibalismo_lesiones": <true o false>,\n'
        '  "descripcion_lesiones": "<detalles de mordeduras de cola, orejas, eritemas o null>",\n'
        '  "evaluacion_clinica_general": "<resumen de conformación, lomo y desarrollo muscular>",\n'
        '  "recomendaciones_manejo": "<recomendación zootécnica nutricional o ambiental>"\n'
        "}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": pure_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                texto_json = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(texto_json)
                return ConceptoVeterinarioSchema(**parsed)
            else:
                print(f"[PorciTech Gemini] Error {resp.status_code} al consultar Gemini: {resp.text}")
                return None
    except Exception as e:
        print(f"[PorciTech Gemini] Excepción al invocar Gemini: {e}")
        return None


# ============================================================================
# 7. ENDPOINTS DE CALIBRACIÓN ARUCO
# ============================================================================
@app.post("/calibrar", response_model=CalibracionRespuesta)
async def calibrar_con_foto(
    file: UploadFile = File(...),
    tamano_marcador_cm: float = Form(DEFAULT_MARKER_SIZE_CM),
    diccionario_id: int = Form(ARUCO_DICCIONARIO_DEFAULT)
):
    """
    Recibe una fotografía del corral VACÍO con el marcador ArUco colocado en el suelo.
    Calcula la escala métrica exacta, la guarda en 'calibracion_camara.json'
    y permite retirar el marcador físico.
    """
    contents = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

    frame = await asyncio.to_thread(decodificar_imagen_ram_sync, contents)
    resultado = detectar_marcador_aruco(frame, tamano_marcador_cm, diccionario_id)

    if resultado is None:
        raise HTTPException(
            status_code=422,
            detail="No se encontró ningún marcador ArUco en la fotografía. "
                   "Asegúrese de buena iluminación y de que el marcador esté plano en el suelo."
        )

    alto, ancho = frame.shape[:2]
    resolucion_str = f"{ancho}x{alto}"

    # Guardar en disco permanentemente
    guardar_calibracion_persistente(
        pixels_per_cm_1280=resultado["pixels_per_cm_base1280"],
        tamano_marcador_cm=tamano_marcador_cm,
        marcador_id=resultado["marcador_id"],
        resolucion=resolucion_str
    )

    _, buffer_jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_jpg).decode('utf-8')}"

    return {
        "status": "ok",
        "mensaje": f"Calibración exitosa. Escala guardada: {resultado['pixels_per_cm_base1280']:.3f} px/cm (base 1280). Puede retirar el marcador del corral.",
        "pixels_per_cm_base1280": round(resultado["pixels_per_cm_base1280"], 3),
        "tamano_marcador_cm": tamano_marcador_cm,
        "marcador_id": resultado["marcador_id"],
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "resolucion_captura": resolucion_str,
        "imagen_anotada_base64": img_b64
    }


@app.post("/calibrar-rtsp", response_model=CalibracionRespuesta)
async def calibrar_desde_rtsp(payload: CalibrarRTSPRequest):
    """
    Captura un fotograma en vivo desde la cámara RTSP apuntando al marcador en el corral
    y persiste la calibración métrica.
    """
    try:
        frame = await asyncio.to_thread(capturar_frame_v380_rtsp_sync, payload.rtsp_url)
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=504, detail=str(e))

    resultado = detectar_marcador_aruco(frame, payload.tamano_marcador_cm, payload.diccionario_aruco)
    if resultado is None:
        raise HTTPException(
            status_code=422,
            detail="No se detectó ningún marcador ArUco en el stream de la cámara RTSP."
        )

    alto, ancho = frame.shape[:2]
    resolucion_str = f"{ancho}x{alto}"

    guardar_calibracion_persistente(
        pixels_per_cm_1280=resultado["pixels_per_cm_base1280"],
        tamano_marcador_cm=payload.tamano_marcador_cm,
        marcador_id=resultado["marcador_id"],
        resolucion=resolucion_str
    )

    _, buffer_jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_jpg).decode('utf-8')}"

    return {
        "status": "ok",
        "mensaje": f"Calibración RTSP completada. Escala guardada: {resultado['pixels_per_cm_base1280']:.3f} px/cm.",
        "pixels_per_cm_base1280": round(resultado["pixels_per_cm_base1280"], 3),
        "tamano_marcador_cm": payload.tamano_marcador_cm,
        "marcador_id": resultado["marcador_id"],
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "resolucion_captura": resolucion_str,
        "imagen_anotada_base64": img_b64
    }


@app.get("/calibracion-actual")
async def obtener_calibracion_actual():
    """Devuelve el estado de la calibración métrica persistida actualmente en disco."""
    calib = cargar_calibracion_persistente()
    if calib:
        return {"status": "calibrado", "datos": calib}
    return {
        "status": "sin_calibrar",
        "mensaje": "No hay archivo de calibración previo. Utilizando valor de fábrica (15.2 px/cm).",
        "pixels_per_cm_defecto": DEFAULT_PIXELS_PER_CM_AT_1280
    }


# ============================================================================
# 8. ENDPOINTS DE PESAJE EN PRODUCCIÓN (SIN MARCADOR REQUERIDO)
# ============================================================================
@app.get("/health")
async def health_check(request: Request):
    modelo_listo = getattr(request.app.state, "model", None) is not None
    calib = cargar_calibracion_persistente()
    return {
        "status": "ok" if modelo_listo else "degraded",
        "service": "vision-artificial",
        "model_loaded": modelo_listo,
        "xgboost_loaded": getattr(request.app.state, "xgb_model", None) is not None,
        "camara_calibrada": calib is not None,
        "calibracion": calib
    }


@app.post("/estimar-peso", response_model=InferenciaVisionResponse)
async def estimar_peso(
    request: Request,
    file: UploadFile = File(...),
    pixels_per_cm: Optional[float] = Form(None),
    conf_threshold: float = Form(0.45),
    validar_bordes: bool = Form(False),
    evaluar_veterinaria: bool = Form(False, description="Activa el diagnóstico clínico multimodal con Gemini"),
    gemini_api_key: Optional[str] = Form(None, description="API Key opcional de Google Gemini")
):
    """
    Recibe la imagen del cerdo (SIN necesidad de marcador en el piso),
    utiliza la calibración previamente guardada en 'calibracion_camara.json'
    y calcula el peso y morfometría completa.
    Opcionalmente invoca a Gemini Multimodal para evaluación de salud animal.
    """
    contents = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")
    if len(contents) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera los 15 MB permitidos.")

    yolo_lock: Optional[asyncio.Lock] = getattr(request.app.state, "yolo_lock", None)
    xgb_model = getattr(request.app.state, "xgb_model", None)

    try:
        frame = await asyncio.to_thread(decodificar_imagen_ram_sync, contents)
        if yolo_lock is not None:
            async with yolo_lock:
                resultado = await asyncio.to_thread(
                    procesar_frame_morfometria_sync,
                    model=request.app.state.model,
                    xgb_model=xgb_model,
                    frame=frame,
                    pixels_per_cm_override=pixels_per_cm,
                    conf_threshold=conf_threshold,
                    validar_bordes=validar_bordes
                )
        else:
            resultado = await asyncio.to_thread(
                procesar_frame_morfometria_sync,
                model=request.app.state.model,
                xgb_model=xgb_model,
                frame=frame,
                pixels_per_cm_override=pixels_per_cm,
                conf_threshold=conf_threshold,
                validar_bordes=validar_bordes
            )

        if evaluar_veterinaria:
            concepto = await evaluar_concepto_veterinario_gemini_async(
                imagen_base64=resultado["imagen_anotada_base64"],
                peso_kg=resultado["peso_estimado_kg"],
                area_cm2=resultado["area_cm2"],
                largo_cm=resultado["largo_cm"],
                ancho_toracico_cm=resultado["ancho_toracico_cm"],
                api_key_override=gemini_api_key
            )
            resultado["concepto_veterinario"] = concepto

        return resultado
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/estimar-peso-rtsp", response_model=InferenciaVisionResponse)
async def estimar_peso_desde_camara_rtsp(
    payload: CapturaRTSPRequest,
    request: Request
):
    """
    Se conecta al stream RTSP de la cámara, captura el fotograma en vivo del cerdo,
    calcula el peso aplicando la calibración guardada y opcionalmente evalúa con Gemini.
    """
    rtsp_url = (payload.rtsp_url or os.getenv("V380_RTSP_URL") or "").strip()
    if not rtsp_url:
        raise HTTPException(
            status_code=400,
            detail="Debe enviar 'rtsp_url' en el JSON o definir V380_RTSP_URL en las variables de entorno."
        )

    try:
        frame = await asyncio.to_thread(capturar_frame_v380_rtsp_sync, rtsp_url)
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=504, detail=str(e))

    yolo_lock: Optional[asyncio.Lock] = getattr(request.app.state, "yolo_lock", None)
    xgb_model = getattr(request.app.state, "xgb_model", None)

    try:
        if yolo_lock is not None:
            async with yolo_lock:
                resultado = await asyncio.to_thread(
                    procesar_frame_morfometria_sync,
                    model=request.app.state.model,
                    xgb_model=xgb_model,
                    frame=frame,
                    pixels_per_cm_override=payload.pixels_per_cm,
                    conf_threshold=payload.conf_threshold,
                    validar_bordes=payload.validar_bordes
                )
        else:
            resultado = await asyncio.to_thread(
                procesar_frame_morfometria_sync,
                model=request.app.state.model,
                xgb_model=xgb_model,
                frame=frame,
                pixels_per_cm_override=payload.pixels_per_cm,
                conf_threshold=payload.conf_threshold,
                validar_bordes=payload.validar_bordes
            )

        if payload.evaluar_veterinaria:
            concepto = await evaluar_concepto_veterinario_gemini_async(
                imagen_base64=resultado["imagen_anotada_base64"],
                peso_kg=resultado["peso_estimado_kg"],
                area_cm2=resultado["area_cm2"],
                largo_cm=resultado["largo_cm"],
                ancho_toracico_cm=resultado["ancho_toracico_cm"],
                api_key_override=payload.gemini_api_key
            )
            resultado["concepto_veterinario"] = concepto

        return resultado
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
