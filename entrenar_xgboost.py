"""
Script de Entrenamiento del Regresor XGBoost para Estimación de Peso Porcino.
Entrena con las 12 características morfométricas extraídas por YOLOv8 + OpenCV:
[area_cm2, largo_cm, ancho_cm, ancho_toracico_cm, ratio_aspecto, hu1..hu7]
y exporta el modelo en formato JSON para inferencia de baja latencia.
"""

import os
import sys
import json
import argparse
import numpy as np

# Columnas esperadas en el dataset
COLUMNAS_FEATURES = [
    "area_cm2",
    "largo_cm",
    "ancho_cm",
    "ancho_toracico_cm",
    "ratio_aspecto",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"
]
COLUMNA_TARGET = "peso_real_kg"


def generar_dataset_sintetico_huxley(num_muestras: int = 300, ruta_csv: str = "dataset_pesos_cerdos.csv"):
    """
    Genera un dataset sintético realista basado en la curva alométrica de Huxley
    (0.0245 * Area^1.052) con dispersión biométrica natural para pruebas iniciales.
    """
    print(f"[XGBoost Trainer] Generando dataset de calibración con {num_muestras} muestras...")
    np.random.seed(42)

    # Simular áreas corporales de cerdos entre 25 kg y 115 kg (etapas de levante y ceba)
    # Área típica entre 1800 cm² y 5200 cm²
    areas_cm2 = np.random.uniform(1800.0, 5200.0, size=num_muestras)

    filas = []
    for area in areas_cm2:
        # Relaciones alométricas proporcionales típicas en cerdos comerciales (Landrace / Large White / Duroc)
        largo = np.sqrt(area * 2.8) + np.random.normal(0, 2.0)
        ancho = (area / largo) + np.random.normal(0, 1.2)
        ancho_toracico = ancho * np.random.uniform(0.88, 0.96)
        ratio_aspecto = largo / ancho

        # Momentos de Hu simulados (escala logarítmica típica del contorno dorsal porcino)
        hu1 = np.random.normal(0.68, 0.04)
        hu2 = np.random.normal(1.72, 0.08)
        hu3 = np.random.normal(8.80, 0.30)
        hu4 = np.random.normal(9.60, 0.35)
        hu5 = np.random.normal(0.0, 0.01)
        hu6 = np.random.normal(10.60, 0.40)
        hu7 = np.random.normal(0.0, 0.01)

        # Peso real de báscula siguiendo la ley de Huxley con 3.5% de variación biológica individual
        peso_base = 0.0245 * (area ** 1.052)
        # Ajuste multivariable: el ancho torácico y el largo influyen en la densidad muscular
        factor_torax = (ancho_toracico / ancho)
        peso_real = round(float(peso_base * factor_torax + np.random.normal(0, 0.8)), 2)

        filas.append([
            round(float(area), 2),
            round(float(largo), 2),
            round(float(ancho), 2),
            round(float(ancho_toracico), 2),
            round(float(ratio_aspecto), 2),
            round(float(hu1), 4),
            round(float(hu2), 4),
            round(float(hu3), 4),
            round(float(hu4), 4),
            round(float(hu5), 4),
            round(float(hu6), 4),
            round(float(hu7), 4),
            peso_real
        ])

    encabezado = ",".join(COLUMNAS_FEATURES + [COLUMNA_TARGET])
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write(encabezado + "\n")
        for fila in filas:
            f.write(",".join(map(str, fila)) + "\n")

    print(f"[XGBoost Trainer] Dataset guardado exitosamente en: {os.path.abspath(ruta_csv)}")
    return ruta_csv


def entrenar_modelo_xgboost(
    ruta_csv: str = "dataset_pesos_cerdos.csv",
    ruta_salida_modelo: str = "modelo_peso_xgboost.json"
):
    try:
        import xgboost as xgb
    except ImportError:
        print("\n[ERROR] La librería 'xgboost' no está instalada en el entorno.")
        print("Para instalarla ejecuta:")
        print("    uv add xgboost\n")
        return False

    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encontró el archivo: {ruta_csv}")
        return False

    print(f"[XGBoost Trainer] Leyendo datos desde: {ruta_csv}...")
    datos = np.genfromtxt(ruta_csv, delimiter=",", skip_header=1)

    X = datos[:, :-1].astype(np.float32)  # 12 características morfométricas
    y = datos[:, -1].astype(np.float32)   # Peso en kg

    # División entrenamiento / prueba (80% / 20%)
    num_total = len(X)
    indices = np.random.permutation(num_total)
    split_idx = int(num_total * 0.8)
    idx_train, idx_test = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Parámetros del modelo nativo de XGBoost (sin dependencias externas)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "seed": 42
    }

    print("[XGBoost Trainer] Entrenando regresor multivariable...")
    bst = xgb.train(params, dtrain, num_boost_round=150)

    # Evaluación de precisión
    predicciones = bst.predict(dtest)
    error_absoluto = np.abs(predicciones - y_test)
    mae = float(np.mean(error_absoluto))
    mape = float(np.mean(error_absoluto / y_test) * 100.0)

    print("---------------------------------------------------------")
    print(f"📊 Métricas de Validación del Modelo:")
    print(f"   • Error Absoluto Medio (MAE): {mae:.2f} kg")
    print(f"   • Error Porcentual Medio (MAPE): {mape:.2f}% (Meta < 5%)")
    print("---------------------------------------------------------")

    # Exportar modelo a JSON nativo
    bst.save_model(ruta_salida_modelo)
    print(f"✅ Modelo guardado en: {os.path.abspath(ruta_salida_modelo)}")
    print("[INFO] 'visionArtificial.py' detectará este archivo automáticamente al iniciar.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenador de Modelo XGBoost para Estimación de Peso")
    parser.add_argument("--sintetico", action="store_true", help="Genera datos sintéticos basados en Huxley para calibración inicial")
    parser.add_argument("--muestras", type=int, default=300, help="Número de muestras a generar si se usa --sintetico")
    parser.add_argument("--csv", type=str, default="dataset_pesos_cerdos.csv", help="Ruta al archivo CSV con datos")
    parser.add_argument("--salida", type=str, default="modelo_peso_xgboost.json", help="Ruta de guardado del modelo JSON")
    args = parser.parse_args()

    if args.sintetico or not os.path.exists(args.csv):
        generar_dataset_sintetico_huxley(num_muestras=args.muestras, ruta_csv=args.csv)

    entrenar_modelo_xgboost(ruta_csv=args.csv, ruta_salida_modelo=args.salida)
