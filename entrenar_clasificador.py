"""
Entrena un clasificador simple (regresion logistica) sobre features.csv
(generado por recolectar_features.py y etiquetado a mano).

Por que leave-one-video-out y no un train/test split al azar:
con pocos videos, varios pares candidatos salen del MISMO video. Si
mezclas esos pares al azar entre train y test, el modelo puede "memorizar"
patrones de ESE video puntual en vez de generalizar -> metricas
infladas y falsas. Leave-one-video-out entrena con todos los videos
menos uno, y evalua en el que quedo afuera, repitiendo para cada video.
Es la validacion correcta dado que tenes pocos videos y muchos pares
por video.

USO:
    python entrenar_clasificador.py --csv features.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "dist_min_ratio", "delta_acercamiento", "delta_separacion",
    "vel_i_antes", "vel_j_antes", "vel_max_antes",
    "ratio_frenazo_min", "angulo_max",
    "n_frames_antes", "n_frames_despues",
]


def leave_one_video_out(df):
    videos = df["video"].unique()
    resultados = []
    y_true_total, y_pred_total = [], []

    for video_test in videos:
        train = df[df["video"] != video_test]
        test = df[df["video"] == video_test]
        if test["label"].nunique() < 1 or train["label"].nunique() < 2:
            continue  # no se puede entrenar/evaluar sin ambas clases

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURES])
        X_test = scaler.transform(test[FEATURES])

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train, train["label"])

        y_pred = clf.predict(X_test)
        y_true_total.extend(test["label"].tolist())
        y_pred_total.extend(y_pred.tolist())

        resultados.append({
            "video": video_test,
            "n_pares": len(test),
            "n_positivos_reales": int(test["label"].sum()),
            "n_positivos_predichos": int(y_pred.sum()),
        })

    return pd.DataFrame(resultados), np.array(y_true_total), np.array(y_pred_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features.csv")
    ap.add_argument("--salida-modelo", default="clasificador_choques.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=FEATURES + ["label"])
    df["label"] = df["label"].astype(int)

    print(f"Total de pares candidatos etiquetados: {len(df)}")
    print(f"Positivos (choque real): {df['label'].sum()}  |  Negativos: {(df['label']==0).sum()}")
    print(f"Videos distintos: {df['video'].nunique()}\n")

    print("=== Validacion leave-one-video-out ===")
    detalle, y_true, y_pred = leave_one_video_out(df)
    print(detalle.to_string(index=False))

    if len(y_true) > 0:
        print(f"\nPrecision: {precision_score(y_true, y_pred, zero_division=0):.2f}")
        print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.2f}")
        print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.2f}")
        print(f"Matriz de confusion:\n{confusion_matrix(y_true, y_pred)}")

    # Modelo final: entrenado con TODOS los datos disponibles, para uso en produccion
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(df[FEATURES])
    clf_final = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf_final.fit(X_final, df["label"])

    joblib.dump({"modelo": clf_final, "scaler": scaler_final, "features": FEATURES}, args.salida_modelo)
    print(f"\nModelo final guardado en {args.salida_modelo}")

    coefs = pd.Series(clf_final.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print("\nImportancia de cada feature (coeficiente, ya estandarizado):")
    print(coefs.to_string())


if __name__ == "__main__":
    main()
