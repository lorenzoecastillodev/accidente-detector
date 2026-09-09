import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict, deque
from deteccion_pares import evaluar_par, centro, tamano_promedio, punto_inferior, VENTANA_ANTES, VENTANA_DESPUES

try:
    from ultralytics.trackers.basetrack import BaseTrack
except ImportError:
    BaseTrack = None

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
st.title("🚦 Detector de Accidentes de Tráfico en Tiempo Real")

# ------------------------------------------------------------------
# Helpers de render. Los colores (verde/rojo/azul/gris) se aplican via
# HTML inline dentro de st.markdown; el look de "tarjeta blanca con
# borde" lo da st.container(border=True), que ya respeta el tema claro
# definido en .streamlit/config.toml (necesario para que se vea como
# el mockup en vez del tema oscuro por defecto de Streamlit).
# ------------------------------------------------------------------

def render_estado(camara_ok, ia_ok, n_accidentes, confianza_pct):
    camara_val = '<span style="color:#16a34a;font-weight:600;">Conectada</span>' if camara_ok \
        else '<span style="color:#9ca3af;font-weight:600;">Sin video</span>'
    ia_val = '<span style="color:#16a34a;font-weight:600;">Activa</span>' if ia_ok \
        else '<span style="color:#9ca3af;font-weight:600;">Inactiva</span>'
    acc_color = "#dc2626" if n_accidentes > 0 else "#9ca3af"
    conf_val = f'<span style="color:#2563eb;font-weight:600;">{confianza_pct}</span>' if confianza_pct != "—" \
        else '<span style="color:#9ca3af;font-weight:600;">—</span>'
    return f"""
    <div style="display:flex;justify-content:space-between;padding:8px 0;">
        <span>📷 <b>Cámara:</b></span> {camara_val}
    </div>
    <div style="display:flex;justify-content:space-between;padding:8px 0;">
        <span>🧠 <b>Detección IA:</b></span> {ia_val}
    </div>
    <div style="display:flex;justify-content:space-between;padding:8px 0;">
        <span>⚠️ <b>Accidentes detectados:</b></span> <span style="color:{acc_color};font-weight:600;">{n_accidentes}</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:8px 0;">
        <span>🛡️ <b>Confianza del modelo:</b></span> {conf_val}
    </div>
    """


def render_info(lineas):
    if not lineas:
        contenido = ("El sistema analiza el tráfico en tiempo real y detecta accidentes "
                      "automáticamente. Acá van a aparecer las anomalías (frenazo, giro "
                      "brusco, etc.) que provocaron cada alerta.")
    else:
        contenido = "<br><br>".join(lineas)
    return f"""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                padding:14px 16px;color:#1e3a8a;font-size:0.92rem;line-height:1.5;">
        {contenido}
    </div>
    """


col_carga, col_estado, col_info = st.columns([1.1, 1, 1])

with col_carga:
    with st.container(border=True):
        st.subheader("1. Cargar video")
        video_file = st.file_uploader(
            "Arrastra tu video aquí o seleccioná un archivo",
            type=["mp4", "avi", "mov"],
        )
        st.caption("Formatos soportados: MP4, AVI, MOV")

with col_estado:
    with st.container(border=True):
        st.subheader("Estado del sistema")
        estado_ph = st.empty()

with col_info:
    with st.container(border=True):
        st.subheader("Información")
        info_ph = st.empty()

estado_ph.markdown(render_estado(False, False, 0, "—"), unsafe_allow_html=True)
info_ph.markdown(render_info([]), unsafe_allow_html=True)

st.markdown("")
st.subheader("2. Vista en tiempo real")
video_card = st.container(border=True)
with video_card:
    live_badge_ph = st.empty()
    stframe = st.empty()
    alerta_placeholder = st.empty()
    status_bar_ph = st.empty()

if video_file is not None:
    with st.spinner("Cargando modelo..."):
        model = cargar_modelo()

    # Reset explicito del contador de IDs de ByteTrack. 'model' esta
    # cacheado con @st.cache_resource y persiste entre distintos videos
    # subidos en la misma sesion; en varias versiones de ultralytics el
    # contador de IDs de tracking es un atributo de CLASE compartido en
    # todo el proceso, no algo ligado a la instancia del modelo. Sin este
    # reset, subir un segundo video en la misma sesion puede arrastrar
    # IDs/estado del video anterior aunque persist=False se use en el
    # primer frame.
    if BaseTrack is not None:
        BaseTrack.reset_id()

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    TAMANO_VENTANA = VENTANA_ANTES + VENTANA_DESPUES + 1

    posiciones_historial = defaultdict(list)
    posiciones_giro_historial = defaultdict(list)
    tamanos_historial = defaultdict(list)
    frame_buffer = deque(maxlen=TAMANO_VENTANA)
    alertas_activas = {}

    # Persistentes para todo el video actual (no se resetean con "vida")
    total_accidentes = 0
    historial_razones = []  # mas reciente primero
    confianza_actual = "—"

    estado_ph.markdown(render_estado(True, True, 0, confianza_actual), unsafe_allow_html=True)
    live_badge_ph.markdown(
        '<div style="background:#111827;color:#fff;padding:6px 14px;border-radius:6px;'
        'display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:0.85rem;">'
        '<span style="width:8px;height:8px;border-radius:50%;background:#ef4444;"></span> EN VIVO</div>',
        unsafe_allow_html=True,
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # persist=False en el primer frame de CADA video: resetea el tracker
        # de ByteTrack (IDs, buffers internos) para que no arrastre estado
        # del video anterior procesado en la misma sesion de Streamlit.
        # 'model' esta cacheado con @st.cache_resource y persiste entre
        # uploads, pero su tracker interno no se reinicia solo.
        results = model.track(frame, persist=(frame_count > 1), imgsz=640, verbose=False,
                               classes=[2, 3, 5, 7], conf=0.3, tracker="bytetrack_custom.yaml")
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        ids_presentes = []

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(ids)):
                vid = int(ids[i])
                pos = centro(coords[i])
                pos_giro = punto_inferior(coords[i])   # NUEVO
                tam = tamano_promedio(coords[i])
                posiciones_historial[vid].append((frame_count, pos))
                posiciones_giro_historial[vid].append((frame_count, pos_giro))   # NUEVO
                tamanos_historial[vid].append((frame_count, tam))
                ids_presentes.append(vid)

            if len(confs) > 0:
                confianza_actual = f"{confs.mean()*100:.0f}%"
                estado_ph.markdown(
                    render_estado(True, True, total_accidentes, confianza_actual),
                    unsafe_allow_html=True,
                )

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
                    hist_i_giro = [(f, p) for f, p in posiciones_giro_historial[id_i]]
                    hist_j_giro = [(f, p) for f, p in posiciones_giro_historial[id_j]]

                    evento = evaluar_par(id_i, id_j, serie, tam_prom, hist_i, hist_j, hist_i_giro, hist_j_giro)
                    if evento:
                        alertas_activas[par] = {"razones": evento["razones"], "vida": 45}

                        # Registro PERSISTENTE: no se borra hasta el proximo video
                        total_accidentes += 1
                        historial_razones.insert(0, (frame_min_local, par, evento["razones"]))

                        estado_ph.markdown(
                            render_estado(True, True, total_accidentes, confianza_actual),
                            unsafe_allow_html=True,
                        )

                        lineas = [
                            f'🚨 <b>Frame {f}</b> (IDs {p}): {", ".join(r)}'
                            for f, p, r in historial_razones
                        ]
                        info_ph.markdown(render_info(lineas), unsafe_allow_html=True)

        for par in list(alertas_activas.keys()):
            alertas_activas[par]["vida"] -= 1
            if alertas_activas[par]["vida"] <= 0:
                del alertas_activas[par]

        # Banner rojo TRANSITORIO: solo mientras la alerta esta "viva" (45 frames).
        # Distinto del panel de Informacion, que queda fijo con el historial completo.
        if alertas_activas:
            textos = [f"IDs {par}: {' + '.join(info['razones'])}" for par, info in alertas_activas.items()]
            alerta_placeholder.error("🚨 POSIBLE ACCIDENTE DETECTADO — " + " | ".join(textos))
        else:
            alerta_placeholder.empty()

        if frame_count % 2 == 0:
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

        status_bar_ph.markdown(
            f'<div style="display:flex;justify-content:space-between;padding-top:8px;'
            f'color:#374151;font-size:0.85rem;">'
            f'<span><span style="width:8px;height:8px;border-radius:50%;background:#22c55e;'
            f'display:inline-block;margin-right:6px;"></span>Analizando video en tiempo real...</span>'
            f'<span><b>FPS:</b> {fps_video:.0f} &nbsp;&nbsp; <b>Resolución:</b> {ancho}x{alto}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    cap.release()
    try:
        os.unlink(video_path)
    except PermissionError:
        pass

    live_badge_ph.empty()
    status_bar_ph.markdown(
        f'<div style="display:flex;justify-content:space-between;padding-top:8px;'
        f'color:#374151;font-size:0.85rem;">'
        f'<span><span style="width:8px;height:8px;border-radius:50%;background:#9ca3af;'
        f'display:inline-block;margin-right:6px;"></span>Video finalizado</span>'
        f'<span><b>FPS:</b> {fps_video:.0f} &nbsp;&nbsp; <b>Resolución:</b> {ancho}x{alto}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.success("¡Procesamiento terminado!")