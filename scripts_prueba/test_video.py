from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("video-calle.mp4", save=True, stream=True, imgsz=320)

for r in results:
    print("Frame procesado")

print("¡Listo! Revisa la carpeta 'runs' para ver el video procesado.")