import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict, deque
from deteccion_pares import evaluar_par, centro, tamano_promedio, VENTANA_ANTES, VENTANA_DESPUES

@st.cache_resource
def cargar_modelo():
    try:
        if not os.path.exists("yolov8s_openvino_model"):
            modelo_base = YOLO("yolov8s.pt")
            modelo_base.export(format="openvino")
        return YOLO("yolov8s_openvino_model/")
    except Exception:
        return YOLO("yolov8s.pt")

st.set_page_config(page_title="Detector de Accidentes", layout="wide")
st.title("🚦 Sistema de Detección de Accidentes de Tráfico")

model = cargar_modelo()

video_file = st.file_uploader("Sube un video de tráfico", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    video_path = tfile.name

    st.write("Procesando video...")
    stframe = st.empty()
    alerta_placeholder = st.empty()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    TAMANO_VENTANA = VENTANA_ANTES + VENTANA_DESPUES + 1

    posiciones_historial = defaultdict(list)
    tamanos_historial = defaultdict(list)
    frame_buffer = deque(maxlen=TAMANO_VENTANA)
    pares_evaluados = set()
    alertas_activas = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        results = model.track(frame, persist=True, imgsz=640, verbose=False,
                               classes=[2, 3, 5, 7], conf=0.3, tracker="bytetrack_custom.yaml")
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        ids_presentes = []

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            for i in range(len(ids)):
                vid = int(ids[i])
                pos = centro(coords[i])
                tam = tamano_promedio(coords[i])
                posiciones_historial[vid].append((frame_count, pos))
                tamanos_historial[vid].append((frame_count, tam))
                ids_presentes.append(vid)

        frame_buffer.append((frame_count, ids_presentes))

        if len(frame_buffer) == TAMANO_VENTANA:
            frame_candidato = frame_buffer[VENTANA_ANTES][0]
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

                    hist_i = [(f, p) for f, p in posiciones_historial[id_i]]
                    hist_j = [(f, p) for f, p in posiciones_historial[id_j]]

                    evento = evaluar_par(id_i, id_j, serie, tam_prom, hist_i, hist_j)
                    if evento:
                        alertas_activas[par] = {"razones": evento["razones"], "vida": 45}

        for par in list(alertas_activas.keys()):
            alertas_activas[par]["vida"] -= 1
            if alertas_activas[par]["vida"] <= 0:
                del alertas_activas[par]

        if alertas_activas:
            textos = [f"IDs {par}: {' + '.join(info['razones'])}" for par, info in alertas_activas.items()]
            alerta_placeholder.error("🚨 POSIBLE ACCIDENTE DETECTADO — " + " | ".join(textos))
        else:
            alerta_placeholder.empty()

        if frame_count % 2 == 0:
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    try:
        os.unlink(video_path)
    except PermissionError:
        pass
    st.success("¡Procesamiento terminado!")