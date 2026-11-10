"""
Generador de Marcadores Fiduciales ArUco para Calibración Métrica Cenital.
Genera un archivo PNG listo para imprimir a escala real (ej. 10 cm x 10 cm o 15 cm x 15 cm).
"""

import os
import cv2
import numpy as np


def generar_marcador_aruco(
    marker_id: int = 0,
    diccionario_tipo: int = cv2.aruco.DICT_4X4_50,
    tamano_pixeles: int = 800,
    margen_borde: int = 100,
    ruta_salida: str = "marcador_aruco_id0_10cm.png"
):
    # 1. Obtener el diccionario ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(diccionario_tipo)

    # 2. Generar el patrón binario del marcador
    marcador_puro = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tamano_pixeles)

    # 3. Agregar un margen blanco alrededor (Quiet Zone para facilitar detección en el corral)
    ancho_total = tamano_pixeles + (2 * margen_borde)
    lienzo = np.ones((ancho_total, ancho_total), dtype=np.uint8) * 255
    lienzo[margen_borde:margen_borde + tamano_pixeles, margen_borde:margen_borde + tamano_pixeles] = marcador_puro

    # 4. Agregar rótulo informativo
    cv2.putText(
        lienzo,
        f"ArUco DICT_4X4_50 - ID: {marker_id} | Calibracion PorciTech",
        (margen_borde, margen_borde - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    # 5. Guardar la imagen
    cv2.imwrite(ruta_salida, lienzo)
    print(f"Marcador ArUco ID {marker_id} generado exitosamente en: {os.path.abspath(ruta_salida)}")


if __name__ == "__main__":
    generar_marcador_aruco(
        marker_id=0,
        tamano_pixeles=800,
        margen_borde=100,
        ruta_salida="marcador_aruco_id0_10cm.png"
    )
