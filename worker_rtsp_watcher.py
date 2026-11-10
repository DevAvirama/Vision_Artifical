"""
Daemon de Monitoreo Autónomo y Pesaje Continuo por RTSP.
Se conecta al stream de la cámara del corral en vivo, detecta automáticamente
cuando un cerdo entra y está centrado bajo la cámara (validar_bordes=True),
estima su peso con XGBoost/ArUco, guarda evidencia fotográfica y registra
los pesajes en un archivo CSV cronológico ('pesajes_diarios.csv').

Uso:
    python worker_rtsp_watcher.py --rtsp "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
"""

import os
import sys
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime, timezone

import httpx

# Carpeta donde se almacenan las fotos anotadas de cada pesaje
CARPETA_EVIDENCIAS = Path("evidencias_pesajes")
ARCHIVO_HISTORICO_CSV = Path("pesajes_diarios.csv")


def iniciar_vigilancia_rtsp(
    rtsp_url: str,
    intervalo_segundos: float = 3.0,
    umbral_confianza: float = 0.50,
    cooldown_segundos: float = 15.0,
    servidor_url: str = "http://localhost:8001/estimar-peso-rtsp",
    backend_webhook_url: str = None
):
    CARPETA_EVIDENCIAS.mkdir(exist_ok=True)

    # Inicializar CSV si no existe
    if not ARCHIVO_HISTORICO_CSV.exists():
        with open(ARCHIVO_HISTORICO_CSV, "w", encoding="utf-8") as f:
            f.write(
                "fecha_utc,peso_kg,confianza_ia,area_cm2,largo_cm,ancho_torax_cm,algoritmo,escala_ppc,foto_evidencia\n"
            )

    print("=========================================================")
    print("👁️  PORCITECH VISION: VIGILANCIA AUTÓNOMA RTSP ACTIVA")
    print("=========================================================")
    print(f" • Cámara RTSP:         {rtsp_url}")
    print(f" • Microservicio:       {servidor_url}")
    print(f" • Intervalo muestreo:  Cada {intervalo_segundos}s")
    print(f" • Cooldown entre cerdos: {cooldown_segundos}s")
    print(f" • Registro histórico:  {ARCHIVO_HISTORICO_CSV.resolve()}")
    print(f" • Carpeta evidencias:  {CARPETA_EVIDENCIAS.resolve()}")
    print("=========================================================\n")
    print("Esperando que un ejemplar ingrese al corral...")

    client = httpx.Client(timeout=15.0)
    ultimo_pesaje_tiempo = 0.0

    try:
        while True:
            ahora = time.time()

            # Evitar registrar múltiples veces al mismo cerdo que se queda parado
            if (ahora - ultimo_pesaje_tiempo) < cooldown_segundos:
                time.sleep(intervalo_segundos)
                continue

            payload = {
                "rtsp_url": rtsp_url,
                "conf_threshold": umbral_confianza,
                "validar_bordes": True  # Excluye capturas donde el cerdo toque el borde
            }

            try:
                resp = client.post(servidor_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    peso = data["peso_estimado_kg"]
                    conf = data["confianza"]
                    area = data["area_cm2"]
                    largo = data["largo_cm"]
                    torax = data["ancho_toracico_cm"]
                    algoritmo = data["algoritmo_peso"]
                    ppc = data["pixels_per_cm_utilizado"]
                    fecha_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    fecha_iso = datetime.now(timezone.utc).isoformat()

                    # Guardar imagen anotada
                    nombre_foto = f"pesaje_{fecha_str}_{peso:.1f}kg.jpg"
                    ruta_foto = CARPETA_EVIDENCIAS / nombre_foto
                    b64_str = data["imagen_anotada_base64"].split(",", 1)[1]
                    with open(ruta_foto, "wb") as f_img:
                        f_img.write(base64.b64decode(b64_str))

                    # Guardar en histórico CSV
                    with open(ARCHIVO_HISTORICO_CSV, "a", encoding="utf-8") as f_csv:
                        f_csv.write(
                            f"{fecha_iso},{peso},{conf:.3f},{area},{largo},{torax},{algoritmo},{ppc},{nombre_foto}\n"
                        )

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] 🐷 ¡NUEVO PESAJE REGISTRADO! "
                        f"Peso: {peso:.2f} kg | Conf: {conf*100:.1f}% | Área: {area:.1f} cm² | "
                        f"Foto: {nombre_foto}"
                    )

                    # Enviar a backend central si está configurado
                    if backend_webhook_url:
                        try:
                            client.post(backend_webhook_url, json={
                                "peso_kg": peso,
                                "confianza": conf,
                                "area_cm2": area,
                                "largo_cm": largo,
                                "ancho_toracico_cm": torax,
                                "fecha_registro": fecha_iso,
                                "escala_ppc": ppc
                            }, timeout=5.0)
                        except Exception as e:
                            print(f"[Aviso] No se pudo enviar webhook a backend: {e}")

                    ultimo_pesaje_tiempo = time.time()
                elif resp.status_code == 422:
                    # 422 indica simplemente que no hay cerdo o está cortado en el borde
                    pass
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Aviso: HTTP {resp.status_code}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Reconectando stream... ({e})")

            time.sleep(intervalo_segundos)
    except KeyboardInterrupt:
        print("\n\n[INFO] Vigilancia RTSP detenida por el operador.")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daemon de Vigilancia y Pesaje Continuo RTSP")
    parser.add_argument("--rtsp", type=str, required=True, help="URL RTSP de la cámara (ej. rtsp://user:pass@IP:554/live)")
    parser.add_argument("--intervalo", type=float, default=2.5, help="Intervalo en segundos entre chequeos (default: 2.5s)")
    parser.add_argument("--cooldown", type=float, default=15.0, help="Segundos de espera para no re-pesar al mismo cerdo (default: 15s)")
    parser.add_argument("--conf", type=float, default=0.50, help="Umbral de confianza mínima YOLO (default: 0.50)")
    parser.add_argument("--servidor", type=str, default="http://localhost:8001/estimar-peso-rtsp", help="URL del endpoint de pesaje RTSP")
    parser.add_argument("--webhook", type=str, default=None, help="URL opcional de webhook en backend para registrar pesaje")
    args = parser.parse_args()

    iniciar_vigilancia_rtsp(
        rtsp_url=args.rtsp,
        intervalo_segundos=args.intervalo,
        umbral_confianza=args.conf,
        cooldown_segundos=args.cooldown,
        servidor_url=args.servidor,
        backend_webhook_url=args.webhook
    )
