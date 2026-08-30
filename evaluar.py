import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
from deteccion_pares import evaluar_par, centro, tamano_promedio
from deteccion_pares import evaluar_par, centro, tamano_promedio, punto_inferior

def extraer_trayectorias(model, video_path, tracker_path, imgsz=640):
    cap = cv2.VideoCapture(video_path)
    posiciones = defaultdict(dict)
    posiciones_giro = defaultdict(dict)
    tamanos = defaultdict(dict)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        results = model.track(frame, persist=True, imgsz=imgsz, verbose=False,
                               classes=[2, 3, 5, 7], conf=0.3, tracker=tracker_path)
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            for i in range(len(ids)):
                vid = int(ids[i])
                posiciones[vid][frame_count] = centro(coords[i])
                posiciones_giro[vid][frame_count] = punto_inferior(coords[i])
                tamanos[vid][frame_count] = tamano_promedio(coords[i])
    cap.release()
    return posiciones, tamanos, posiciones_giro

def detectar_todos_los_pares(posiciones, tamanos, posiciones_giro):
    ids = list(posiciones.keys())
    eventos = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_i, id_j = ids[i], ids[j]
            frames_comunes = sorted(set(posiciones[id_i].keys()) & set(posiciones[id_j].keys()))
            if len(frames_comunes) < 10:
                continue
            serie = [(f, ((posiciones[id_i][f][0]-posiciones[id_j][f][0])**2 +
                          (posiciones[id_i][f][1]-posiciones[id_j][f][1])**2) ** 0.5)
                     for f in frames_comunes]
            dist_min = min(d for f, d in serie)
            frame_min = [f for f, d in serie if d == dist_min][0]
            tam_prom = (tamanos[id_i].get(frame_min, 50) + tamanos[id_j].get(frame_min, 50)) / 2

            hist_i = sorted(posiciones[id_i].items())
            hist_j = sorted(posiciones[id_j].items())
            hist_i_giro = sorted(posiciones_giro[id_i].items())
            hist_j_giro = sorted(posiciones_giro[id_j].items())

            evento = evaluar_par(id_i, id_j, serie, tam_prom, hist_i, hist_j, hist_i_giro, hist_j_giro)
            if evento:
                eventos.append(evento)
    return eventos

set_prueba = [
    {"video": "dataset_prueba/clip-choque-1.mp4", "tiene_accidente": True, "segundo_esperado": 2.3, "cuenta": True},
    {"video": "dataset_prueba/video-calle.mp4", "tiene_accidente": False, "segundo_esperado": None, "cuenta": True},
    {"video": "dataset_prueba/clip-noche-rapido.mp4", "tiene_accidente": True, "segundo_esperado": 4.2, "cuenta": True},
    {"video": "dataset_prueba/Video Project 2.mp4", "tiene_accidente": True, "segundo_esperado": 4.9, "cuenta": True},
    {"video": "dataset_prueba/video-trampa-mercado.mp4", "tiene_accidente": False, "segundo_esperado": None, "cuenta": True},
    {"video": "dataset_prueba/clip-choque-2-v2.mp4", "tiene_accidente": True, "segundo_esperado": 2.1, "cuenta": False},
    {"video": "dataset_prueba/video-frenazo-normal.mp4", "tiene_accidente": False, "segundo_esperado": None, "cuenta": True},
    {"video": "dataset_prueba/video-giro-normal.mp4", "tiene_accidente": False, "segundo_esperado": None, "cuenta": True},
{"video": "dataset_prueba/video-interseccion-amsterdam.mp4", "tiene_accidente": False, "segundo_esperado": None, "cuenta": True},
]

modelo_path = "yolov8s_openvino_model/" if os.path.exists("yolov8s_openvino_model") else "yolov8s.pt"
tracker_path = "bytetrack_custom.yaml"

print(f"Usando modelo: {modelo_path}\n")

resultados = []

for caso in set_prueba:
    model = YOLO(modelo_path)
    cap = cv2.VideoCapture(caso["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    posiciones, tamanos, posiciones_giro = extraer_trayectorias(model, caso["video"], tracker_path)
    eventos = detectar_todos_los_pares(posiciones, tamanos, posiciones_giro)

    nombre = os.path.basename(caso["video"])
    detecto = len(eventos) > 0

    print(f"\n=== {nombre} (esperado: {'SI' if caso['tiene_accidente'] else 'NO'}) ===")
    if eventos:
        for e in eventos:
            print(f"  Segundo {round(e['frame']/fps,2)}s (frame {e['frame']}): par={e['par']} razones={e['razones']}")
    else:
        print("  Sin eventos detectados")

    if caso["tiene_accidente"] and detecto and caso["segundo_esperado"] is not None:
        margen = 1.5
        acierto_tiempo = any(abs(e["frame"]/fps - caso["segundo_esperado"]) <= margen for e in eventos)
        print(f"  Esperado en ~{caso['segundo_esperado']}s -> Timing correcto: {'SI' if acierto_tiempo else 'NO'}")

    resultados.append({
        "nombre": nombre,
        "esperado": caso["tiene_accidente"],
        "detecto": detecto,
        "cuenta": caso.get("cuenta", True)
    })

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSION (solo videos que cuentan)\n")

TP = sum(1 for r in resultados if r["cuenta"] and r["esperado"] and r["detecto"])
FN = sum(1 for r in resultados if r["cuenta"] and r["esperado"] and not r["detecto"])
FP = sum(1 for r in resultados if r["cuenta"] and not r["esperado"] and r["detecto"])
TN = sum(1 for r in resultados if r["cuenta"] and not r["esperado"] and not r["detecto"])

print(f"Verdaderos Positivos (TP): {TP}  -> choques reales, detectados")
print(f"Falsos Negativos   (FN): {FN}  -> choques reales, NO detectados")
print(f"Falsos Positivos   (FP): {FP}  -> sin choque, pero disparo alerta")
print(f"Verdaderos Negativos (TN): {TN}  -> sin choque, correctamente sin alerta")

precision = TP / (TP + FP) if (TP + FP) > 0 else None
recall = TP / (TP + FN) if (TP + FN) > 0 else None
f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
fpr = FP / (FP + TN) if (FP + TN) > 0 else None

print(f"\nPrecision: {precision*100:.0f}%" if precision is not None else "\nPrecision: N/A")
print(f"Recall:    {recall*100:.0f}%" if recall is not None else "Recall: N/A")
print(f"F1-score:  {f1*100:.0f}%" if f1 is not None else "F1-score: N/A")
print(f"Tasa de Falsos Positivos: {fpr*100:.0f}%" if fpr is not None else "Tasa de Falsos Positivos: N/A")

excluidos = [r["nombre"] for r in resultados if not r["cuenta"]]
if excluidos:
    print(f"\nVideos excluidos de la metrica (casos atipicos documentados): {excluidos}")