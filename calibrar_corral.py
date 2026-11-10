"""
Script de Calibración del Corral con Marcador ArUco.
Permite calibrar la escala métrica de la cámara en un solo comando,
ya sea enviando una fotografía local o conectándose directamente por RTSP.
"""

import sys
import base64
import argparse
from pathlib import Path
import httpx


def calibrar(
    ruta_foto: str = None,
    rtsp_url: str = None,
    tamano_cm: float = 10.0,
    servidor_url: str = "http://localhost:8001"
):
    client = httpx.Client(timeout=30.0)

    if rtsp_url:
        print(f"Conectando a stream RTSP para calibración: {rtsp_url}...")
        url = f"{servidor_url}/calibrar-rtsp"
        payload = {"rtsp_url": rtsp_url, "tamano_marcador_cm": tamano_cm}
        try:
            resp = client.post(url, json=payload)
        except Exception as e:
            print(f"[ERROR] No se pudo conectar con el servidor: {e}")
            sys.exit(1)
    elif ruta_foto:
        path = Path(ruta_foto)
        if not path.exists():
            print(f"[ERROR] No se encontró el archivo: {ruta_foto}")
            sys.exit(1)

        print(f"Enviando fotografía '{path.name}' para calibración ArUco...")
        url = f"{servidor_url}/calibrar"
        with open(path, "rb") as f:
            archivos = {"file": (path.name, f, "image/jpeg")}
            datos = {"tamano_marcador_cm": str(tamano_cm)}
            try:
                resp = client.post(url, files=archivos, data=datos)
            except Exception as e:
                print(f"[ERROR] No se pudo conectar con el servidor: {e}")
                sys.exit(1)
    else:
        print("[ERROR] Debes proporcionar --foto o --rtsp")
        sys.exit(1)

    if resp.status_code == 200:
        data = resp.json()
        print("\n=========================================================")
        print("✅ CALIBRACIÓN MÉTRICA EXITOSA")
        print("=========================================================")
        print(f" • Escala Guardada:       {data['pixels_per_cm_base1280']} px/cm (base 1280)")
        print(f" • Marcador ArUco ID:     {data['marcador_id']}")
        print(f" • Tamaño Real Marcador:  {data['tamano_marcador_cm']} cm")
        print(f" • Resolución de Toma:    {data['resolucion_captura']}")
        print(f" • Fecha de Calibración:  {data['fecha_calibracion']}")
        print("=========================================================")
        print("💾 La escala quedó guardada permanentemente en: calibracion_camara.json")
        print("👉 ¡YA PUEDES RETIRAR EL MARCADOR FÍSICO DEL CORRAL!\n")

        # Guardar imagen con el marcador detectado en cian
        if "imagen_anotada_base64" in data and data["imagen_anotada_base64"]:
            b64_str = data["imagen_anotada_base64"].split(",", 1)[1]
            with open("calibracion_confirmada.jpg", "wb") as f_out:
                f_out.write(base64.b64decode(b64_str))
            print("🖼️  Imagen de confirmación guardada en: calibracion_confirmada.jpg\n")
    else:
        print(f"\n[ERROR {resp.status_code}]: {resp.text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrador de Corral PorciTech con Marcador ArUco")
    parser.add_argument("--foto", type=str, default=None, help="Ruta a la foto del corral con el marcador")
    parser.add_argument("--rtsp", type=str, default=None, help="URL RTSP de la cámara IP del corral")
    parser.add_argument("--cm", type=float, default=10.0, help="Tamaño real del lado del marcador en cm (default: 10.0)")
    parser.add_argument("--servidor", type=str, default="http://localhost:8001", help="URL base del microservicio")
    args = parser.parse_args()

    calibrar(
        ruta_foto=args.foto,
        rtsp_url=args.rtsp,
        tamano_cm=args.cm,
        servidor_url=args.servidor
    )
