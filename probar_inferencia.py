"""
Script de prueba rápida para inferencia de peso porcino desde la terminal.
Envía una imagen al microservicio en ejecución, imprime las métricas morfométricas
y guarda la imagen anotada con el peso sobre el lomo en 'resultado_anotado.jpg'.
"""

import sys
import base64
import httpx
from pathlib import Path


def probar_imagen(ruta_imagen: str, url_servidor: str = "http://localhost:8001/estimar-peso"):
    path = Path(ruta_imagen)
    if not path.exists():
        print(f"[ERROR] No se encontró el archivo: {ruta_imagen}")
        sys.exit(1)

    print(f"Enviando '{path.name}' al microservicio ({url_servidor})...")
    with open(path, "rb") as f:
        archivos = {"file": (path.name, f, "image/jpeg")}
        datos = {"conf_threshold": "0.45", "validar_bordes": "false"}

        try:
            resp = httpx.post(url_servidor, files=archivos, data=datos, timeout=30.0)
        except Exception as e:
            print(f"[ERROR] No se pudo conectar con el microservicio: {e}")
            print("Asegúrate de tener corriendo: uv run python main.py")
            sys.exit(1)

    if resp.status_code == 200:
        data = resp.json()
        print("\n=========================================================")
        print("📊 RESULTADOS DE LA ESTIMACIÓN:")
        print("=========================================================")
        print(f" • Peso Estimado:      {data['peso_estimado_kg']} kg")
        print(f" • Algoritmo Usado:    {data['algoritmo_peso']}")
        print(f" • Método Calibración: {data['metodo_calibracion']}")
        print(f" • Escala Aplicada:    {data['pixels_per_cm_utilizado']} px/cm")
        print(f" • Área Dorsal:        {data['area_cm2']} cm²")
        print(f" • Longitud Corporal:  {data['largo_cm']} cm")
        print(f" • Ancho Máximo:       {data['ancho_cm']} cm")
        print(f" • Ancho Torácico:     {data['ancho_toracico_cm']} cm")
        print(f" • Confianza IA:       {data['confianza'] * 100:.1f}%")
        print("=========================================================")

        # Guardar imagen anotada
        b64_str = data["imagen_anotada_base64"].split(",", 1)[1]
        with open("resultado_anotado.jpg", "wb") as f_out:
            f_out.write(base64.b64decode(b64_str))
        print("🖼️  Imagen con silueta y etiqueta guardada en: resultado_anotado.jpg\n")
    else:
        print(f"[ERROR {resp.status_code}]: {resp.text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        probar_imagen(sys.argv[1])
    else:
        print("Uso: python probar_inferencia.py ruta/a/tu_foto_cerdo.jpg")
