"""
Recorre una carpeta de videos y guarda en un CSV una fila por
cada par candidato que paso los gates fisicos (acercandose + movimiento_real),
sin aplicar el sistema de puntos viejo. Esto sirve para juntar EJEMPLOS
NEGATIVOS de trafico pesado (que hoy tu score confunde con choque) y
EJEMPLOS POSITIVOS reales, y despues entrenar un clasificador con
entrenar_clasificador.py.

USO:
    python recolectar_features.py --carpeta dataset_prueba --salida features.csv
    python recolectar_features.py --carpeta dataset_trafico_pesado --salida features.csv --append

Despues de correr esto, abris el CSV y llenas a mano la columna "label":
    1 = este par fue el choque real (compara "frame" contra el segundo
        esperado del video, si lo sabes)
    0 = no hubo choque en este video, o este par no es el choque
        (aunque haya pasado los gates)

NOTA 1: el nombre guardado en la columna "video" del CSV incluye el nombre
de la carpeta de origen (ej: "dataset_tudat_normal/v1.mov"), no solo el
nombre del archivo, para evitar colisiones cuando dos carpetas distintas
tienen archivos con el mismo nombre.

NOTA 2: se calcula el FPS real de cada video y se lo pasa a
calcular_features_par(), igual que ya se hace en evaluar.py y app.py. Sin
esto, las velocidades y ventanas de tiempo quedan calibradas para ~30 FPS
y no son comparables con videos a otro FPS (ver deteccion_pares.py).
"""
import argparse
import csv
import os
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

from deteccion_pares import (
    calcular_features_par, centro, punto_inferior, tamano_promedio,
    VENTANA_ANTES_SEG, VENTANA_DESPUES_SEG, ventana_en_frames,
)

try:
    from ultralytics.trackers.basetrack import BaseTrack
except ImportError:
    BaseTrack = None

CAMPOS_CSV = [
    "video", "id_i", "id_j", "frame",
    "dist_min", "dist_min_ratio", "delta_acercamiento", "delta_separacion", "separandose",
    "vel_i_antes", "vel_j_antes", "vel_i_desp", "vel_j_desp", "vel_max_antes",
    "ratio_frenazo_i", "ratio_frenazo_j", "ratio_frenazo_min",
    "angulo_i", "angulo_j", "angulo_max",
    "n_frames_antes", "n_frames_despues",
    "label",  # a completar a mano: 1 = choque real, 0 = no
]


def recolectar_video(model, video_path, tracker_path, nombre_video, imgsz=640):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    if BaseTrack is not None:
        BaseTrack.reset_id()

    # FPS real de ESTE video - todas las ventanas de tiempo y velocidades
    # se escalan a este valor, en vez de asumir 30 FPS fijo.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
        print(f"  AVISO: no se pudo leer el FPS de {nombre_video}, usando 30 por defecto")

    ventana_antes_f = ventana_en_frames(VENTANA_ANTES_SEG, fps)
    ventana_despues_f = ventana_en_frames(VENTANA_DESPUES_SEG, fps)
    tamano_ventana_video = ventana_antes_f + ventana_despues_f + 1
    dedup_ventana_f = ventana_en_frames(0.5, fps)  # antes hardcodeado en 15 frames (@30fps = 0.5s)

    posiciones_historial = defaultdict(list)
    posiciones_giro_historial = defaultdict(list)
    tamanos_historial = defaultdict(list)
    frame_buffer = deque(maxlen=tamano_ventana_video)
    ya_visto = set()  # evita duplicar el mismo par decenas de veces frame a frame
    filas = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        results = model.track(frame, persist=True, imgsz=imgsz, verbose=False,
                               classes=[2, 3, 5, 7], conf=0.3, tracker=tracker_path)
        boxes = results[0].boxes
        ids_presentes = []

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            for i in range(len(ids)):
                vid = int(ids[i])
                posiciones_historial[vid].append((frame_count, centro(coords[i])))
                posiciones_giro_historial[vid].append((frame_count, punto_inferior(coords[i])))
                tamanos_historial[vid].append((frame_count, tamano_promedio(coords[i])))
                ids_presentes.append(vid)

        frame_buffer.append((frame_count, ids_presentes))

        if len(frame_buffer) == tamano_ventana_video:
            ids_en_candidato = frame_buffer[ventana_antes_f][1]
            for i in range(len(ids_en_candidato)):
                for j in range(i + 1, len(ids_en_candidato)):
                    id_i, id_j = ids_en_candidato[i], ids_en_candidato[j]
                    par = tuple(sorted((id_i, id_j)))

                    frames_ventana = [f for f, _ in frame_buffer]
                    f_min_v, f_max_v = min(frames_ventana), max(frames_ventana)
                    pos_i = {f: p for f, p in posiciones_historial[id_i] if f_min_v <= f <= f_max_v}
                    pos_j = {f: p for f, p in posiciones_historial[id_j] if f_min_v <= f <= f_max_v}
                    frames_comunes = sorted(set(pos_i.keys()) & set(pos_j.keys()))
                    if len(frames_comunes) < 10:
                        continue

                    serie = [(f, ((pos_i[f][0]-pos_j[f][0])**2 + (pos_i[f][1]-pos_j[f][1])**2) ** 0.5)
                             for f in frames_comunes]

                    tam_i = dict(tamanos_historial[id_i])
                    tam_j = dict(tamanos_historial[id_j])
                    dist_min_val = min(d for f, d in serie)
                    frame_min_local = [f for f, d in serie if d == dist_min_val][0]
                    tam_prom = (tam_i.get(frame_min_local, 50) + tam_j.get(frame_min_local, 50)) / 2

                    hist_i = list(posiciones_historial[id_i])
                    hist_j = list(posiciones_historial[id_j])
                    hist_i_giro = list(posiciones_giro_historial[id_i])
                    hist_j_giro = list(posiciones_giro_historial[id_j])

                    feats = calcular_features_par(id_i, id_j, serie, tam_prom, hist_i, hist_j, hist_i_giro, hist_j_giro, fps=fps)
                    if feats is None or not (feats["acercandose"] and feats["movimiento_real"]):
                        continue

                    clave_dedup = (par, feats["frame"] // dedup_ventana_f)
                    if clave_dedup in ya_visto:
                        continue
                    ya_visto.add(clave_dedup)

                    fila = {c: feats.get(c) for c in CAMPOS_CSV if c not in ("video", "label")}
                    fila["video"] = nombre_video
                    fila["id_i"], fila["id_j"] = id_i, id_j
                    fila["frame"] = feats["frame"]
                    fila["label"] = ""  # completar a mano despues
                    filas.append(fila)

    cap.release()
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", required=True, help="Carpeta con videos a procesar")
    ap.add_argument("--salida", default="features.csv")
    ap.add_argument("--modelo", default="yolov8s.pt")
    ap.add_argument("--tracker", default="bytetrack_custom.yaml")
    ap.add_argument("--append", action="store_true", help="Agregar al CSV existente en vez de sobreescribir")
    args = ap.parse_args()

    videos = [f for f in os.listdir(args.carpeta) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    print(f"Encontrados {len(videos)} videos en {args.carpeta}")

    modo = "a" if args.append and os.path.exists(args.salida) else "w"
    escribir_header = modo == "w"

    prefijo_carpeta = os.path.basename(os.path.normpath(args.carpeta))

    with open(args.salida, modo, newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CAMPOS_CSV)
        if escribir_header:
            writer.writeheader()

        for nombre in videos:
            print(f"Procesando {nombre}...")
            model = YOLO(args.modelo)
            nombre_para_csv = f"{prefijo_carpeta}/{nombre}"
            filas = recolectar_video(model, os.path.join(args.carpeta, nombre), args.tracker, nombre_para_csv)
            print(f"  {len(filas)} pares candidatos encontrados")
            for fila in filas:
                writer.writerow(fila)
            f_out.flush()  # fuerza a escribir a disco YA, no esperar a que termine todo el script.
            os.fsync(f_out.fileno())  # y confirma que el sistema operativo lo bajo a disco de verdad.

    print(f"\nListo. Revisa {args.salida} y completa la columna 'label' a mano:")
    print("  1 = este par es el choque real del video")
    print("  0 = no hubo choque, o este par no es el choque (falso positivo del gate)")


if __name__ == "__main__":
    main()