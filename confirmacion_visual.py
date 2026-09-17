"""
Confirmacion visual OPCIONAL e INFORMATIVA usando un modelo YOLOv8
propio (yolov8n), entrenado localmente en Google Colab sobre el
dataset publico "accident-detection-model-knp0m" de Roboflow
(~3250 imagenes, licencia CC BY 4.0). Metricas del modelo en SU
dataset de entrenamiento (no el dataset de prueba de este proyecto):
Precision 81.3%, Recall 69.3%, mAP50 73.1%.

Que es y que NO es:
- Corre 100% LOCAL. No requiere cuenta, API key, ni internet en
  tiempo de ejecucion - solo la libreria ultralytics, que el proyecto
  ya usa. La unica dependencia externa es descargar el archivo de
  pesos UNA VEZ (ver README) desde el GitHub Release "modelo-visual-v1".
- Es una SEGUNDA OPINION basada en como SE VE la imagen (auto danado,
  posicion anormal), completamente independiente de deteccion_pares.py
  (que decide por MOVIMIENTO: velocidades, angulos, distancias). No
  modifica ni reemplaza esa logica en absoluto.
- Es EXPERIMENTAL: entrenado con datos de terceros, no con videos de
  este proyecto. Tratar sus resultados con escepticismo hasta medir
  su impacto real (ver evaluar.py).
- Si el archivo de pesos no esta presente, todo lo demas del proyecto
  sigue funcionando exactamente igual - esto nunca es un requisito.
"""
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

MODELO_VISUAL_PATH = "accidente_visual_v1.pt"

_modelo_visual = None
_intento_carga = False


def _cargar_modelo_visual():
    global _modelo_visual, _intento_carga
    if _intento_carga:
        return _modelo_visual
    _intento_carga = True

    if YOLO is None:
        return None
    if not os.path.exists(MODELO_VISUAL_PATH):
        print(f"[confirmacion_visual] No se encontro '{MODELO_VISUAL_PATH}' en la "
              f"raiz del proyecto. Descargalo del GitHub Release 'modelo-visual-v1' "
              f"si queres activar esta confirmacion opcional (ver README). El "
              f"sistema principal de deteccion funciona igual sin esto.")
        return None

    try:
        _modelo_visual = YOLO(MODELO_VISUAL_PATH)
    except Exception as e:
        print(f"[confirmacion_visual] No se pudo cargar el modelo: {e}")
        _modelo_visual = None
    return _modelo_visual


def confirmar_visualmente(frame_bgr):
    """
    Corre el modelo visual local sobre un frame (array BGR, el mismo
    formato que devuelve cv2.VideoCapture / cv2.read).

    Devuelve un porcentaje de confianza (0-100) de que la escena
    contenga un accidente segun este modelo, o None si el modelo no
    esta disponible (pesos no descargados, error de carga, etc.) -
    tratar None como "sin dato", nunca como "no es un accidente".
    """
    modelo = _cargar_modelo_visual()
    if modelo is None:
        return None

    try:
        resultados = modelo.predict(frame_bgr, verbose=False, conf=0.25)
        boxes = resultados[0].boxes
        if boxes is None or len(boxes) == 0:
            return 0.0
        confs = boxes.conf.cpu().numpy()
        return float(confs.max()) * 100
    except Exception as e:
        print(f"[confirmacion_visual] Error al correr inferencia: {e}")
        return None