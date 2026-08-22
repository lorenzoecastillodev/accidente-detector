import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict, deque

def overlap_relativo(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    min_area = min(a1, a2)
    return inter / min_area if min_area > 0 else 0

def centro(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calcular_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])

def magnitud(v):
    return (v[0]**2 + v[1]**2) ** 0.5

@st.cache_resource
def cargar_modelo():
    import os
    try:
        if not os.path.exists("yolov8s_openvino_model"):
            modelo_base = YOLO("yolov8s.pt")
            modelo_base.export(format="openvino")
        modelo = YOLO("yolov8s_openvino_model/")
        st.info("Usando OpenVINO (optimizado para CPU Intel)")
        return modelo
    except Exception:
        st.info("OpenVINO no disponible, usando modelo estándar")
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

    UMBRAL_OVERLAP = 0.15
    OVERLAP_PREVIO_MAX = 0.05
    FRAMES_CONFIRMACION = 3
    HISTORIAL_LEN = 8
    VENTANA_GIRO = 3
    CAIDA_VELOCIDAD = 0.35
    VELOCIDAD_MINIMA = 5
    ANGULO_CAMBIO_BRUSCO = 45
    GRACIA_FRAMES = 8
    IMGSZ = 640

    historial_posiciones = defaultdict(lambda: deque(maxlen=HISTORIAL_LEN))
    contador_frenazo = defaultdict(int)
    contador_giro = defaultdict(int)
    contador_overlap = defaultdict(int)
    overlap_anterior = defaultdict(float)
    primer_frame_visto = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        results = model.track(frame, persist=True, imgsz=IMGSZ, verbose=False,
                               classes=[2, 3, 5, 7], conf=0.3, tracker="bytetrack_custom.yaml")
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        anomalia_trayectoria = set()
        etiquetas_debug = {}
        velocidad_actual = {}

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()

            for i in range(len(ids)):
                vid = int(ids[i])
                if vid not in primer_frame_visto:
                    primer_frame_visto[vid] = frame_count

                cx, cy = centro(coords[i])
                historial_posiciones[vid].append((cx, cy))

                hist = historial_posiciones[vid]
                frames_desde_aparicion = frame_count - primer_frame_visto[vid]
                velocidad_actual[vid] = 0

                if len(hist) >= HISTORIAL_LEN and frames_desde_aparicion >= GRACIA_FRAMES:
                    p_inicio = hist[0]
                    p_medio = hist[len(hist) // 2]
                    p_fin = hist[-1]

                    vec_antes = calcular_vector(p_inicio, p_medio)
                    vec_despues = calcular_vector(p_medio, p_fin)
                    vel_antes = magnitud(vec_antes)
                    vel_despues = magnitud(vec_despues)
                    velocidad_actual[vid] = max(vel_antes, vel_despues)

                    frenazo = vel_antes > VELOCIDAD_MINIMA and vel_despues < vel_antes * CAIDA_VELOCIDAD

                    p_giro_ini = hist[-VENTANA_GIRO * 2] if len(hist) >= VENTANA_GIRO * 2 else hist[0]
                    p_giro_medio = hist[-VENTANA_GIRO] if len(hist) >= VENTANA_GIRO else hist[len(hist) // 2]
                    p_giro_fin = hist[-1]

                    vec_g_antes = calcular_vector(p_giro_ini, p_giro_medio)
                    vec_g_despues = calcular_vector(p_giro_medio, p_giro_fin)
                    vel_g_antes = magnitud(vec_g_antes)
                    vel_g_despues = magnitud(vec_g_despues)

                    cambio_direccion = False
                    if vel_g_antes > VELOCIDAD_MINIMA and vel_g_despues > VELOCIDAD_MINIMA:
                        cos_ang = (vec_g_antes[0]*vec_g_despues[0] + vec_g_antes[1]*vec_g_despues[1]) / (vel_g_antes * vel_g_despues)
                        cos_ang = max(-1, min(1, cos_ang))
                        angulo = np.degrees(np.arccos(cos_ang))
                        cambio_direccion = angulo > ANGULO_CAMBIO_BRUSCO

                    contador_frenazo[vid] = contador_frenazo[vid] + 1 if frenazo else 0
                    contador_giro[vid] = contador_giro[vid] + 1 if cambio_direccion else 0

                    if contador_frenazo[vid] >= FRAMES_CONFIRMACION or contador_giro[vid] >= FRAMES_CONFIRMACION:
                        anomalia_trayectoria.add(vid)
                        razon = []
                        if contador_frenazo[vid] >= FRAMES_CONFIRMACION:
                            razon.append("FRENAZO")
                        if contador_giro[vid] >= FRAMES_CONFIRMACION:
                            razon.append("GIRO BRUSCO")
                        etiquetas_debug[vid] = " + ".join(razon)

            ids_con_anomalia = set(anomalia_trayectoria)

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id_i, id_j = int(ids[i]), int(ids[j])
                    par = tuple(sorted((id_i, id_j)))
                    ov = overlap_relativo(coords[i], coords[j])

                    overlap_nuevo = ov > UMBRAL_OVERLAP and overlap_anterior[par] < OVERLAP_PREVIO_MAX
                    algun_involucrado_anomalo = id_i in anomalia_trayectoria or id_j in anomalia_trayectoria

                    if overlap_nuevo and algun_involucrado_anomalo:
                        ids_con_anomalia.add(id_i)
                        ids_con_anomalia.add(id_j)
                        etiquetas_debug[id_i] = etiquetas_debug.get(id_i, "") + " + OVERLAP"
                        etiquetas_debug[id_j] = etiquetas_debug.get(id_j, "") + " + OVERLAP"

                    overlap_anterior[par] = ov

            for i in range(len(ids)):
                vid = int(ids[i])
                if vid in etiquetas_debug:
                    x1, y1, x2, y2 = coords[i].astype(int)
                    cv2.putText(annotated_frame, etiquetas_debug[vid], (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            ids_con_anomalia = set()

        if ids_con_anomalia:
            alerta_placeholder.error(f"🚨 POSIBLE ACCIDENTE DETECTADO — vehiculo(s) ID: {sorted(ids_con_anomalia)}")
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