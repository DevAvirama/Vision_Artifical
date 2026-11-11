"""
Script de Evaluación por Lotes (Batch Evaluation) para el Dataset de Visión Porcina.
Procesa un directorio de imágenes de cerdos, compara el peso estimado contra el peso
real de báscula y genera métricas estadísticas completas (MAE, MAPE, RMSE, R²).

Uso:
    python evaluar_lote.py --dir ruta/a/fotos/ --csv pesos_reales.csv
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import httpx
except ImportError:
    print("[ERROR] Requiere httpx: uv add httpx")
    sys.exit(1)


def evaluar_directorio(
    directorio_fotos: str,
    archivo_pesos_csv: str = None,
    escala_ppc_override: float = None,
    url_api: str = "http://localhost:8001/estimar-peso",
    salida_reporte_csv: str = "reporte_evaluacion_lote.csv"
):
    dir_path = Path(directorio_fotos)
    if not dir_path.exists():
        print(f"[ERROR] El directorio '{directorio_fotos}' no existe.")
        sys.exit(1)

    # 1. Cargar pesos reales si se proporciona el CSV
    pesos_reales = {}
    if archivo_pesos_csv and os.path.exists(archivo_pesos_csv):
        print(f"Cargando pesos reales desde: {archivo_pesos_csv}...")
        with open(archivo_pesos_csv, "r", encoding="utf-8") as f:
            for linea in f:
                partes = linea.strip().split(",")
                if len(partes) >= 2:
                    try:
                        nombre_archivo = partes[0].strip()
                        peso = float(partes[1].strip())
                        pesos_reales[nombre_archivo] = peso
                    except ValueError:
                        continue
        print(f"Se cargaron {len(pesos_reales)} pesos de referencia.")

    # 2. Buscar imágenes válidas
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imagenes = [p for p in dir_path.iterdir() if p.suffix.lower() in extensiones]
    if not imagenes:
        print(f"[ERROR] No se encontraron imágenes en: {directorio_fotos}")
        sys.exit(1)

    print(f"\nIniciando evaluación de {len(imagenes)} imágenes contra {url_api}...")
    print("=" * 70)

    filas_reporte = []
    errores_absolutos = []
    errores_porcentuales = []
    pesos_y_true = []
    pesos_y_pred = []

    client = httpx.Client(timeout=45.0)

    for idx, img_path in enumerate(imagenes, 1):
        nombre = img_path.name
        peso_real = pesos_reales.get(nombre, None)

        with open(img_path, "rb") as f:
            archivos = {"file": (nombre, f, "image/jpeg")}
            data = {"conf_threshold": "0.45", "validar_bordes": "false"}
            if escala_ppc_override and escala_ppc_override > 0:
                data["pixels_per_cm"] = str(escala_ppc_override)

            try:
                resp = client.post(url_api, files=archivos, data=data)
            except Exception as e:
                print(f"[{idx}/{len(imagenes)}] ❌ {nombre}: Error de conexión ({e})")
                continue

        if resp.status_code == 200:
            res = resp.json()
            peso_pred = res["peso_estimado_kg"]
            area = res["area_cm2"]
            largo = res["largo_cm"]
            ancho_t = res["ancho_toracico_cm"]
            ppc = res["pixels_per_cm_utilizado"]
            algoritmo = res["algoritmo_peso"]

            if peso_real is not None:
                err_abs = abs(peso_pred - peso_real)
                err_pct = (err_abs / peso_real) * 100.0
                errores_absolutos.append(err_abs)
                errores_porcentuales.append(err_pct)
                pesos_y_true.append(peso_real)
                pesos_y_pred.append(peso_pred)

                print(
                    f"[{idx}/{len(imagenes)}] ✅ {nombre:22s} | Real: {peso_real:6.2f} kg | "
                    f"Pred: {peso_pred:6.2f} kg | Error: {err_abs:5.2f} kg ({err_pct:5.2f}%)"
                )
                filas_reporte.append([
                    nombre, peso_real, peso_pred, round(err_abs, 2), round(err_pct, 2),
                    area, largo, ancho_t, ppc, algoritmo
                ])
            else:
                print(
                    f"[{idx}/{len(imagenes)}] ℹ️  {nombre:22s} | Pred: {peso_pred:6.2f} kg | "
                    f"Área: {area:7.1f} cm² | Algoritmo: {algoritmo}"
                )
                filas_reporte.append([
                    nombre, "N/A", peso_pred, "N/A", "N/A", area, largo, ancho_t, ppc, algoritmo
                ])
        else:
            print(f"[{idx}/{len(imagenes)}] ⚠️ {nombre}: {resp.status_code} - {resp.text}")

    client.close()

    # 3. Guardar reporte CSV detallado
    with open(salida_reporte_csv, "w", encoding="utf-8") as f_out:
        f_out.write("archivo,peso_real_kg,peso_predicho_kg,error_abs_kg,error_pct,area_cm2,largo_cm,ancho_torax_cm,ppc_utilizado,algoritmo\n")
        for fila in filas_reporte:
            f_out.write(",".join(map(str, fila)) + "\n")

    print("\n" + "=" * 70)
    print(f"📄 Reporte individual guardado en: {os.path.abspath(salida_reporte_csv)}")

    # 4. Estadísticas consolidadas si hubo pesos de referencia
    if errores_absolutos:
        mae = float(np.mean(errores_absolutos))
        mape = float(np.mean(errores_porcentuales))
        rmse = float(np.sqrt(np.mean(np.array(errores_absolutos) ** 2)))
        max_err = float(np.max(errores_absolutos))

        # Cálculo de R² (Coeficiente de Determinación)
        y_t = np.array(pesos_y_true)
        y_p = np.array(pesos_y_pred)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-9))

        print("📊 MÉTRICAS GLOBALES DEL MODELO EN EL LOTE:")
        print("---------------------------------------------------------")
        print(f" • Muestras Evaluadas:             {len(errores_absolutos)}")
        print(f" • Error Absoluto Medio (MAE):      {mae:.2f} kg")
        print(f" • Error Porcentual Medio (MAPE):   {mape:.2f}% (Meta < 5%)")
        print(f" • Raíz del Error Cuadrático (RMSE):{rmse:.2f} kg")
        print(f" • Error Máximo Registrado:        {max_err:.2f} kg")
        print(f" • Coeficiente de Determinación (R²):{r2:.4f}")
        print("---------------------------------------------------------")
        if mape < 5.0:
            print("🌟 ¡EXCELENTE! El modelo cumple holgadamente con los estándares de la industria (<5%).")
        else:
            print("💡 Se recomienda reentrenar XGBoost con los datos morfométricos de este lote.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluador por Lotes de Pesaje Porcino")
    parser.add_argument("--dir", type=str, required=True, help="Ruta al directorio que contiene las fotos de cerdos")
    parser.add_argument("--csv", type=str, default=None, help="Ruta al archivo CSV con columnas: nombre_archivo,peso_real_kg")
    parser.add_argument("--ppc", type=float, default=None, help="Escala px/cm override si todas las fotos son a altura fija (ej. 6.6)")
    parser.add_argument("--salida", type=str, default="reporte_evaluacion_lote.csv", help="Ruta del CSV de salida con los resultados")
    args = parser.parse_args()

    evaluar_directorio(
        directorio_fotos=args.dir,
        archivo_pesos_csv=args.csv,
        escala_ppc_override=args.ppc,
        salida_reporte_csv=args.salida
    )
