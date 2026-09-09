"""
Script de diagnostico: corre el MISMO video 3 veces seguidas con el
mismo codigo (procesar_video de evaluar.py), para confirmar si la
discrepancia app.py/evaluar.py se debe a no-determinismo del motor de
inferencia (OpenVINO multi-thread en CPU) en vez de a un bug de logica.

Si las 3 corridas dan resultados DISTINTOS entre si, confirma
no-determinismo: el problema no esta en el codigo, esta en que el
sistema no es reproducible run-to-run tal como esta configurado.

Si las 3 corridas dan resultados IDENTICOS, el no-determinismo queda
descartado y hay que seguir buscando la diferencia real en otro lado
(por ejemplo, en como Streamlit lee o reescribe el archivo de video).

Uso:
    python test_determinismo.py dataset_prueba/choque-azul.mp4
"""
import sys
import os
from ultralytics import YOLO
from evaluar import procesar_video

if len(sys.argv) < 2:
    print("Uso: python test_determinismo.py <ruta_al_video>")
    sys.exit(1)

video_path = sys.argv[1]
modelo_path = "yolov8s_openvino_model/" if os.path.exists("yolov8s_openvino_model") else "yolov8s.pt"
tracker_path = "bytetrack_custom.yaml"

N_CORRIDAS = 3
resultados_por_corrida = []

for corrida in range(1, N_CORRIDAS + 1):
    # Modelo nuevo cada vez, igual que hace evaluar.py normalmente,
    # para no mezclar esta prueba con el tema de tracker compartido.
    model = YOLO(modelo_path)
    eventos = procesar_video(model, video_path, tracker_path)

    resumen = [(e["frame"], e["par"], tuple(e["razones"])) for e in eventos]
    resultados_por_corrida.append(resumen)

    print(f"\n=== Corrida {corrida} ===")
    if resumen:
        for frame, par, razones in resumen:
            print(f"  Frame {frame}: par={par} razones={list(razones)}")
    else:
        print("  Sin eventos")

print("\n" + "=" * 60)
if all(r == resultados_por_corrida[0] for r in resultados_por_corrida):
    print("RESULTADO: Las 3 corridas dieron IDENTICO resultado.")
    print("-> No-determinismo descartado. La diferencia con app.py")
    print("   esta en otro lado (revisar lectura de video, ejecucion")
    print("   frame por frame en Streamlit, etc.).")
else:
    print("RESULTADO: Las corridas dieron resultados DISTINTOS entre si.")
    print("-> Confirmado: hay no-determinismo en la inferencia (probable")
    print("   OpenVINO multi-thread). Esto explica la discrepancia con")
    print("   app.py sin que sea un bug de logica. Ver recomendacion de")
    print("   fix en el mensaje de Claude.")
