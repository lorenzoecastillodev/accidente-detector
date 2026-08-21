from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track("video-calle.mp4", save=True, imgsz=320, persist=True, stream=True)

for r in results:
    print("Frame procesado")

print("¡Listo! Revisa la carpeta 'runs' para ver el video con tracking.")