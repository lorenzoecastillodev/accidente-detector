from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("¡Listo! Revisa la carpeta 'runs' para ver el resultado.")