import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO

app = FastAPI(title="Microservicio de Visión PorciTech")

# Cargar el modelo de segmentación
# (puedes usar 'best.pt' si es tu modelo entrenado o 'yolov8n-seg.pt')
print("Cargando modelo de IA...")
model = YOLO('best.pt')

# Factor de calibración (Píxeles a Kilos)
FACTOR_PIXELES_A_KILOS = 0.0005 

@app.get("/health")
def health_check():
    """Endpoint para verificar que el servicio está activo."""
    return {"status": "ok", "service": "vision-artificial"}

@app.post("/estimar-peso")
async def estimar_peso(file: UploadFile = File(...)):
    """Recibe una imagen, segmenta el animal y calcula su peso estimado."""
    # 1. Validar y leer los bytes de la imagen enviada
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen enviada.")

    # 2. Ejecutar inferencia con YOLO
    resultados = model(frame, verbose=False)

    detecciones = []

    for resultado in resultados:
        if resultado.masks is not None:
            for mask in resultado.masks.xy:
                contorno = np.array(mask, dtype=np.int32)
                area_pixeles = cv2.contourArea(contorno)

                # Filtrar ruido visual menor a 5000 px
                if area_pixeles > 5000:
                    x, y, w, h = cv2.boundingRect(contorno)
                    peso_estimado = area_pixeles * FACTOR_PIXELES_A_KILOS

                    detecciones.append({
                        "area_pixeles": float(area_pixeles),
                        "peso_estimado_kg": round(float(peso_estimado), 2),
                        "bounding_box": {"x": int(x), "y": int(y), "ancho": int(w), "alto": int(h)}
                    })

    if not detecciones:
        return {
            "mensaje": "No se detectó ningún animal con suficiente área en la imagen.",
            "detecciones": []
        }

    return {
        "total_detectados": len(detecciones),
        "detecciones": detecciones
    }
