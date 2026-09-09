import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import os
from deteccion_pares import evaluar_par, centro, tamano_promedio, punto_inferior, VENTANA_ANTES, VENTANA_DESPUES

try:
    from ultralytics.trackers.basetrack import BaseTrack
except ImportError:
    BaseTrack = None
    print("AVISO: no se pudo importar BaseTrack para resetear IDs de tracking "
          "entre videos. Puede que la version de ultralytics instalada tenga "
          "otra ruta de modulo. Revisar si las metricas parecen inconsistentes.")

# Debe coincidir EXACTAMENTE con la logica de app.py (misma "regla de oro" del
# proyecto: cualquier cambio aca debe reflejarse tambien alla, y viceversa).
TAMANO_VENTANA = VENTANA_ANTES + VENTANA_DESPUES + 1
VIDA_ALERTA = 45  # frames que una alerta permanece activa antes de poder re-evaluarse (igual que app.py)


def procesar_video(model, video_path, tracker_path, imgsz=640):
    """
    Replica frame por frame la logica real de app.py:
    - Tracking incremental (no se ven frames futuros).
    - Ventana deslizante de TAMANO_VENTANA frames; se evalua el frame central
      de la ventana (frame_buffer[VENTANA_ANTES]) en cada iteracion.
    - hist_i/hist_j/hist_i_giro/hist_j_giro son el historial ACUMULADO hasta
      el frame actual (causal), igual que en app.py. Solo 'serie' (la
      distancia par a par) se restringe a la ventana actual.
    - Un par que ya disparo alerta se ignora (`continue`) durante VIDA_ALERTA
      frames antes de poder volver a evaluarse.

    Devuelve la lista de eventos, uno por cada vez que un par dispara alerta
    (mismo formato que devuelve evaluar_par).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Reset explicito del contador de IDs de ByteTrack. Crear un YOLO()
    # nuevo por video NO alcanza: en varias versiones de ultralytics, el
    # contador de IDs de tracking es un atributo de CLASE compartido en
    # todo el proceso de Python, no algo ligado a la instancia del modelo.
    # Sin este reset, el video N+1 puede arrancar con IDs ya "usados" y
    # arrastrar comportamiento del video N, aunque el modelo sea nuevo.
    if BaseTrack is not None:
        BaseTrack.reset_id()

    posiciones_historial = defaultdict(list)
    posiciones_giro_historial = defaultdict(list)
    tamanos_historial = defaultdict(list)
    frame_buffer = deque(maxlen=TAMANO_VENTANA)
    alertas_activas = {}
    eventos = []

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
                pos = centro(coords[i])
                pos_giro = punto_inferior(coords[i])
                tam = tamano_promedio(coords[i])
                posiciones_historial[vid].append((frame_count, pos))
                posiciones_giro_historial[vid].append((frame_count, pos_giro))
                tamanos_historial[vid].append((frame_count, tam))
                ids_presentes.append(vid)

        frame_buffer.append((frame_count, ids_presentes))

        if len(frame_buffer) == TAMANO_VENTANA:
            ids_en_candidato = frame_buffer[VENTANA_ANTES][1]

            for i in range(len(ids_en_candidato)):
                for j in range(i + 1, len(ids_en_candidato)):
                    id_i, id_j = ids_en_candidato[i], ids_en_candidato[j]
                    par = tuple(sorted((id_i, id_j)))
                    if par in alertas_activas:
                        continue

                    frames_ventana = [f for f, _ in frame_buffer]
                    f_min_ventana, f_max_ventana = min(frames_ventana), max(frames_ventana)

                    pos_i = {f: p for f, p in posiciones_historial[id_i] if f_min_ventana <= f <= f_max_ventana}
                    pos_j = {f: p for f, p in posiciones_historial[id_j] if f_min_ventana <= f <= f_max_ventana}
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

                    # Historial CAUSAL completo hasta el frame actual (no solo la ventana),
                    # igual que app.py.
                    hist_i = list(posiciones_historial[id_i])
                    hist_j = list(posiciones_historial[id_j])
                    hist_i_giro = list(posiciones_giro_historial[id_i])
                    hist_j_giro = list(posiciones_giro_historial[id_j])

                    evento = evaluar_par(id_i, id_j, serie, tam_prom, hist_i, hist_j, hist_i_giro, hist_j_giro)
                    if evento:
                        alertas_activas[par] = {"razones": evento["razones"], "vida": VIDA_ALERTA}
                        eventos.append(evento)

        for par in list(alertas_activas.keys()):
            alertas_activas[par]["vida"] -= 1
            if alertas_activas[par]["vida"] <= 0:
                del alertas_activas[par]

    cap.release()
    return eventos


# NOTA: todo lo de aca abajo esta envuelto en "if __name__ == '__main__':"
# a proposito. Sin esto, con solo IMPORTAR este archivo desde otro script
# (por ejemplo, para reusar procesar_video()), Python ejecuta el modulo
# completo de punta a punta -> se dispara sin querer la evaluacion de
# los 11 videos como efecto secundario de un simple import.
if __name__ == "__main__":
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
        {"video": "dataset_prueba/choque-puente.mp4", "tiene_accidente": True, "segundo_esperado": 0.6, "cuenta": True},
        {"video": "dataset_prueba/choque-azul.mp4", "tiene_accidente": True, "segundo_esperado": 3.3, "cuenta": True},
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

        eventos = procesar_video(model, caso["video"], tracker_path)

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