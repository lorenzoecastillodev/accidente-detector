import cv2
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import os
from deteccion_pares import (
    evaluar_par, centro, tamano_promedio, punto_inferior,
    VENTANA_ANTES_SEG, VENTANA_DESPUES_SEG, ventana_en_frames,
)
from confirmacion_visual import confirmar_visualmente

try:
    from ultralytics.trackers.basetrack import BaseTrack
except ImportError:
    BaseTrack = None
    print("AVISO: no se pudo importar BaseTrack para resetear IDs de tracking "
          "entre videos. Puede que la version de ultralytics instalada tenga "
          "otra ruta de modulo. Revisar si las metricas parecen inconsistentes.")

# VIDA_ALERTA esta calibrado en segundos ahora (1.5s @ 30fps = 45 frames,
# igual que el valor original) y se escala al FPS real de cada video, igual
# que las demas ventanas de tiempo.
VIDA_ALERTA_SEG = 45 / 30


def procesar_video(model, video_path, tracker_path, imgsz=640):
    """
    Replica frame por frame la logica real de app.py:
    - Tracking incremental (no se ven frames futuros).
    - Ventana deslizante de tamano variable segun el FPS real del video
      (antes era un tamano fijo de frames, calibrado asumiendo ~30 FPS -
      ver deteccion_pares_v2.py para el detalle de por que se cambio esto);
      se evalua el frame central de la ventana en cada iteracion.
    - hist_i/hist_j/hist_i_giro/hist_j_giro son el historial ACUMULADO hasta
      el frame actual (causal), igual que en app.py. Solo 'serie' (la
      distancia par a par) se restringe a la ventana actual.
    - Un par que ya disparo alerta se ignora (`continue`) durante VIDA_ALERTA
      frames antes de poder volver a evaluarse.

    Devuelve una tupla (eventos, fps_procesamiento):
    - eventos: lista de eventos, uno por cada vez que un par dispara alerta
      (mismo formato que devuelve evaluar_par).
    - fps_procesamiento: cuadros por segundo REALES que el pipeline completo
      (YOLO + tracking + evaluar_par) logro procesar en esta corrida. Sirve
      para comparar contra los FPS nativos del video y saber si el sistema
      corre a velocidad de "tiempo real" o mas lento.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    tiempo_inicio = time.time()

    # FPS real del video - de aca en mas todas las ventanas de tiempo y
    # umbrales de velocidad se escalan a este valor, en vez de asumir 30.
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if not fps_video or fps_video <= 0:
        fps_video = 30  # fallback por si el video no reporta FPS valido
        print(f"AVISO: no se pudo leer el FPS de {video_path}, usando 30 por defecto")

    ventana_antes_f = ventana_en_frames(VENTANA_ANTES_SEG, fps_video)
    ventana_despues_f = ventana_en_frames(VENTANA_DESPUES_SEG, fps_video)
    tamano_ventana_video = ventana_antes_f + ventana_despues_f + 1
    vida_alerta_video = ventana_en_frames(VIDA_ALERTA_SEG, fps_video)

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
    frame_buffer = deque(maxlen=tamano_ventana_video)
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

        if len(frame_buffer) == tamano_ventana_video:
            ids_en_candidato = frame_buffer[ventana_antes_f][1]

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

                    evento = evaluar_par(id_i, id_j, serie, tam_prom, hist_i, hist_j, hist_i_giro, hist_j_giro, fps=fps_video)
                    if evento:
                        alertas_activas[par] = {"razones": evento["razones"], "vida": vida_alerta_video}

                        # Confirmacion visual EXTERNA e INFORMATIVA (modelo local
                        # entrenado con datos de terceros). No modifica en nada la
                        # deteccion por movimiento de arriba - solo se guarda en el
                        # evento para poder medir despues si aporta algo real.
                        evento["confianza_visual"] = confirmar_visualmente(frame)

                        eventos.append(evento)

        for par in list(alertas_activas.keys()):
            alertas_activas[par]["vida"] -= 1
            if alertas_activas[par]["vida"] <= 0:
                del alertas_activas[par]

    cap.release()
    tiempo_total = time.time() - tiempo_inicio
    fps_procesamiento = frame_count / tiempo_total if tiempo_total > 0 else 0
    return eventos, fps_procesamiento


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

        eventos, fps_procesamiento = procesar_video(model, caso["video"], tracker_path)

        nombre = os.path.basename(caso["video"])
        detecto = len(eventos) > 0

        print(f"\n=== {nombre} (esperado: {'SI' if caso['tiene_accidente'] else 'NO'}) ===")
        es_tiempo_real = fps_procesamiento >= fps
        print(f"  Velocidad: {fps_procesamiento:.1f} FPS procesados vs {fps:.0f} FPS del video "
              f"-> {'tiempo real' if es_tiempo_real else 'MAS LENTO que tiempo real'}")
        if eventos:
            for e in eventos:
                cv_txt = f", confianza_visual={e['confianza_visual']:.0f}%" if e.get("confianza_visual") is not None else ", confianza_visual=N/A"
                print(f"  Segundo {round(e['frame']/fps,2)}s (frame {e['frame']}): par={e['par']} razones={e['razones']}{cv_txt}")
        else:
            print("  Sin eventos detectados")

        acierto_tiempo = None  # None = no aplica (sin choque, o sin segundo_esperado)
        if caso["tiene_accidente"] and detecto and caso["segundo_esperado"] is not None:
            margen = 1.5
            acierto_tiempo = any(abs(e["frame"]/fps - caso["segundo_esperado"]) <= margen for e in eventos)
            print(f"  Esperado en ~{caso['segundo_esperado']}s -> Timing correcto: {'SI' if acierto_tiempo else 'NO'}")

        confianzas_visuales = [e["confianza_visual"] for e in eventos if e.get("confianza_visual") is not None]
        mejor_confianza_visual = max(confianzas_visuales) if confianzas_visuales else None

        resultados.append({
            "nombre": nombre,
            "esperado": caso["tiene_accidente"],
            "detecto": detecto,
            "timing_correcto": acierto_tiempo,
            "confianza_visual": mejor_confianza_visual,
            "fps_procesamiento": fps_procesamiento,
            "fps_video": fps,
            "cuenta": caso.get("cuenta", True)
        })

    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSION - DETECCION (solo videos que cuentan)\n")
    print("¿Disparo alguna alerta en el video? No exige que el timing sea correcto.\n")

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

    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSION - DETECCION CON TIMING CORRECTO (mas estricta)\n")
    print("Una alerta que dispara tarde o en el momento equivocado NO cuenta")
    print("como acierto aca, aunque si haya 'detectado algo' en el video.\n")

    def acerto_bien(r):
        return r["detecto"] and r["timing_correcto"] is True

    TP_t = sum(1 for r in resultados if r["cuenta"] and r["esperado"] and acerto_bien(r))
    FN_t = sum(1 for r in resultados if r["cuenta"] and r["esperado"] and not acerto_bien(r))
    FP_t = FP
    TN_t = TN

    print(f"Verdaderos Positivos (TP): {TP_t}  -> choques reales, detectados CON timing correcto")
    print(f"Falsos Negativos   (FN): {FN_t}  -> choques reales, no detectados O detectados con timing incorrecto")
    print(f"Falsos Positivos   (FP): {FP_t}  -> sin choque, pero disparo alerta")
    print(f"Verdaderos Negativos (TN): {TN_t}  -> sin choque, correctamente sin alerta")

    precision_t = TP_t / (TP_t + FP_t) if (TP_t + FP_t) > 0 else None
    recall_t = TP_t / (TP_t + FN_t) if (TP_t + FN_t) > 0 else None
    f1_t = (2 * precision_t * recall_t / (precision_t + recall_t)) if precision_t and recall_t and (precision_t + recall_t) > 0 else None

    print(f"\nPrecision (con timing): {precision_t*100:.0f}%" if precision_t is not None else "\nPrecision (con timing): N/A")
    print(f"Recall    (con timing): {recall_t*100:.0f}%" if recall_t is not None else "Recall (con timing): N/A")
    print(f"F1-score  (con timing): {f1_t*100:.0f}%" if f1_t is not None else "F1-score (con timing): N/A")

    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSION - CON CONFIRMACION VISUAL (experimental)\n")
    print("Ademas de detectar por movimiento, exige que el modelo visual local")
    print("(entrenado con datos de terceros, ver confirmacion_visual.py) tambien")
    print("de una confianza >= UMBRAL_CONFIANZA_VISUAL. Si el archivo de pesos")
    print("no esta descargado, confianza_visual siempre es None y esta matriz")
    print("va a verse peor de lo real - no sacar conclusiones sin el modelo cargado.\n")

    UMBRAL_CONFIANZA_VISUAL = 25  # primer valor a probar, NO esta ajustado con evidencia todavia

    def confirma_visualmente(r):
        return r["confianza_visual"] is not None and r["confianza_visual"] >= UMBRAL_CONFIANZA_VISUAL

    TP_v = sum(1 for r in resultados if r["cuenta"] and r["esperado"] and r["detecto"] and confirma_visualmente(r))
    FN_v = sum(1 for r in resultados if r["cuenta"] and r["esperado"] and not (r["detecto"] and confirma_visualmente(r)))
    FP_v = sum(1 for r in resultados if r["cuenta"] and not r["esperado"] and r["detecto"] and confirma_visualmente(r))
    TN_v = sum(1 for r in resultados if r["cuenta"] and not r["esperado"] and not (r["detecto"] and confirma_visualmente(r)))

    print(f"Verdaderos Positivos (TP): {TP_v}")
    print(f"Falsos Negativos   (FN): {FN_v}")
    print(f"Falsos Positivos   (FP): {FP_v}  -> ojo: FP baja si el modelo visual NO confirma un caso que antes era FP")
    print(f"Verdaderos Negativos (TN): {TN_v}")

    precision_v = TP_v / (TP_v + FP_v) if (TP_v + FP_v) > 0 else None
    recall_v = TP_v / (TP_v + FN_v) if (TP_v + FN_v) > 0 else None

    print(f"\nPrecision (con conf. visual): {precision_v*100:.0f}%" if precision_v is not None else "\nPrecision (con conf. visual): N/A")
    print(f"Recall    (con conf. visual): {recall_v*100:.0f}%" if recall_v is not None else "Recall (con conf. visual): N/A")

    excluidos = [r["nombre"] for r in resultados if not r["cuenta"]]
    if excluidos:
        print(f"\nVideos excluidos de la metrica (casos atipicos documentados): {excluidos}")

    print("\n" + "=" * 70)
    print("VELOCIDAD DE PROCESAMIENTO (¿es realmente 'tiempo real'?)\n")
    fps_prom = sum(r["fps_procesamiento"] for r in resultados) / len(resultados)
    videos_lentos = [r["nombre"] for r in resultados if r["fps_procesamiento"] < r["fps_video"]]
    print(f"FPS promedio de procesamiento: {fps_prom:.1f}")
    if videos_lentos:
        print(f"Videos donde el procesamiento fue MAS LENTO que el video fuente ({len(videos_lentos)}/{len(resultados)}):")
        for r in resultados:
            if r["nombre"] in videos_lentos:
                print(f"  - {r['nombre']}: {r['fps_procesamiento']:.1f} FPS procesados vs {r['fps_video']:.0f} FPS del video")
    else:
        print("Todos los videos se procesaron a velocidad de tiempo real o mas rapido.")
