import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Detector de Accidentes", layout="wide")
st.title("🚦 Sistema de Detección de Accidentes de Tráfico")

model = YOLO("yolov8n.pt")

video_file = st.file_uploader("Sube un video de tráfico", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    video_path = tfile.name

    st.write("Procesando video...")
    stframe = st.empty()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    skip_frames = 3

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        results = model.track(frame, persist=True, imgsz=320, verbose=False)
        annotated_frame = results[0].plot()

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    os.unlink(video_path)
    st.success("¡Procesamiento terminado!")