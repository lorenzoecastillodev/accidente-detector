import numpy as np

def distancia(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def centro(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def punto_inferior(box):
    return ((box[0] + box[2]) / 2, box[3])

def tamano_promedio(box):
    return ((box[2]-box[0]) + (box[3]-box[1])) / 2

# ============================================================
# PARÁMETROS CONGELADOS — 2026-08-29 (sin tocar los VALORES, se
# siguen usando como GATES de candidatos). Actualizado el
# 2026-09: se reexpresan en unidades reales (segundos, pixeles
# por segundo) en vez de "frames"/"pixeles por frame", porque
# estaban calibrados implicitamente para videos a ~30 FPS y
# TU-DAT viene a ~55-60 FPS. A 30 FPS exacto el comportamiento
# es IDENTICO al original (mismos numeros, solo reexpresados).
# ============================================================
FPS_REFERENCIA = 30  # el FPS con el que se calibraron estos 7 parametros originalmente

VENTANA_ANTES = 15       # en frames @ 30fps = 0.5 segundos
VENTANA_DESPUES = 15     # en frames @ 30fps = 0.5 segundos
VENTANA_GIRO = 6         # en frames @ 30fps = 0.2 segundos
UMBRAL_VEL_ACERCAMIENTO = 3   # px/frame @ 30fps = 90 px/segundo
UMBRAL_VEL_ALTA = 6           # px/frame @ 30fps = 180 px/segundo
UMBRAL_ANGULO_GIRO = 40       # grados - no depende del FPS, sin cambios
RATIO_CONTACTO = 0.9          # ratio espacial - no depende del FPS, sin cambios

VENTANA_ANTES_SEG = VENTANA_ANTES / FPS_REFERENCIA
VENTANA_DESPUES_SEG = VENTANA_DESPUES / FPS_REFERENCIA
VENTANA_GIRO_SEG = VENTANA_GIRO / FPS_REFERENCIA
UMBRAL_VEL_ACERCAMIENTO_PXSEG = UMBRAL_VEL_ACERCAMIENTO * FPS_REFERENCIA
UMBRAL_VEL_ALTA_PXSEG = UMBRAL_VEL_ALTA * FPS_REFERENCIA


UMBRAL_VEL_FRENAZO_PXSEG = 4 * FPS_REFERENCIA  # antes "4" comparado contra px/frame; ahora en px/segundo


def ventana_en_frames(segundos, fps):
    """Convierte una ventana de tiempo (en segundos) a cantidad de frames
    para un video con el FPS dado. Con fps=FPS_REFERENCIA da exactamente
    los valores originales (15, 15, 6)."""
    return max(1, round(segundos * fps))


def velocidad_y_direccion(posiciones_ordenadas, frame_referencia, ventana=8, fps=FPS_REFERENCIA):
    """
    Devuelve la velocidad en PIXELES POR SEGUNDO (no por frame), y el vector
    de direccion (en pixeles/frame, porque para el angulo entre vectores no
    importa la magnitud, solo la direccion - por eso angulo_entre() no
    necesita tocarse).

    Por que normalizar: un auto a 60 km/h se mueve la mitad de pixeles por
    frame en un video de 60 FPS que en uno de 30 FPS (el doble de frames
    por segundo = la mitad de tiempo entre frames = la mitad de distancia
    recorrida por frame). Sin esto, el mismo choque real se ve "mas lento"
    en un video de mayor FPS, y los umbrales fijos dejan de tener sentido.
    """
    usados = [p for f, p in posiciones_ordenadas if f <= frame_referencia][-ventana:]
    if len(usados) < 2:
        return None, None
    p_ini, p_fin = usados[0], usados[-1]
    df = len(usados) - 1
    vec = ((p_fin[0]-p_ini[0])/df, (p_fin[1]-p_ini[1])/df)
    vel_por_frame = (vec[0]**2 + vec[1]**2) ** 0.5
    vel_por_segundo = vel_por_frame * fps
    return vel_por_segundo, vec


def angulo_entre(v1, v2):
    m1 = (v1[0]**2 + v1[1]**2) ** 0.5
    m2 = (v2[0]**2 + v2[1]**2) ** 0.5
    if m1 == 0 or m2 == 0:
        return 0
    cos_ang = max(-1, min(1, (v1[0]*v2[0] + v1[1]*v2[1]) / (m1*m2)))
    return np.degrees(np.arccos(cos_ang))


# Sistema de puntuación viejo (se mantiene por compatibilidad;
# evaluar_par lo sigue usando para decidir alertas igual que siempre)
PUNTOS_FRENAZO = 2
PUNTOS_GIRO = 2
PUNTOS_SEPARACION = 1
PUNTOS_VELOCIDAD_ALTA = 1
UMBRAL_SCORE = 3


def calcular_features_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro, fps=FPS_REFERENCIA):
    """
    Calcula TODAS las magnitudes continuas de un par candidato, sin aplicar
    ningún umbral de decisión final (solo los gates físicos mínimos: hay
    suficientes frames y hay posibilidad de contacto).

    fps: el FPS real del video de donde viene este par. Se usa para
    escalar las ventanas de tiempo (antes fijas en frames, ahora fijas en
    segundos) y para que las velocidades salgan en pixeles/segundo,
    comparables entre videos de distinto FPS. Con fps=30 (el valor por
    defecto) el resultado es identico al comportamiento original.
    """
    if len(serie_distancias) < 5:
        return None

    ventana_antes_f = ventana_en_frames(VENTANA_ANTES_SEG, fps)
    ventana_despues_f = ventana_en_frames(VENTANA_DESPUES_SEG, fps)
    ventana_giro_f = ventana_en_frames(VENTANA_GIRO_SEG, fps)

    dist_min = min(d for f, d in serie_distancias)
    frame_min = [f for f, d in serie_distancias if d == dist_min][0]

    if dist_min >= tam_prom_en_min * RATIO_CONTACTO:
        return None

    dists_antes = [d for f, d in serie_distancias if frame_min - ventana_antes_f <= f < frame_min]
    dists_despues = [d for f, d in serie_distancias if frame_min < f <= frame_min + ventana_despues_f]

    acercandose = len(dists_antes) >= 3 and dists_antes[0] > dists_antes[-1] + 5
    separandose = len(dists_despues) >= 3 and dists_despues[-1] > dists_despues[0] + 5

    delta_acercamiento = (dists_antes[0] - dists_antes[-1]) if len(dists_antes) >= 3 else 0.0
    delta_separacion = (dists_despues[-1] - dists_despues[0]) if len(dists_despues) >= 3 else 0.0

    vel_i_antes, _ = velocidad_y_direccion(hist_i, frame_min - 2, fps=fps)
    vel_i_desp, _ = velocidad_y_direccion(hist_i, frame_min + ventana_despues_f, fps=fps)
    vel_j_antes, _ = velocidad_y_direccion(hist_j, frame_min - 2, fps=fps)
    vel_j_desp, _ = velocidad_y_direccion(hist_j, frame_min + ventana_despues_f, fps=fps)

    def ratio_frenazo(v_antes, v_desp):
        if v_antes and v_desp is not None and v_antes > 0:
            return v_desp / v_antes
        return None

    ratio_frenazo_i = ratio_frenazo(vel_i_antes, vel_i_desp)
    ratio_frenazo_j = ratio_frenazo(vel_j_antes, vel_j_desp)

    _, vec_i_giro_antes = velocidad_y_direccion(hist_i_giro, frame_min + ventana_giro_f, ventana=ventana_giro_f, fps=fps)
    _, vec_i_giro_desp = velocidad_y_direccion(hist_i_giro, frame_min + ventana_giro_f * 2, ventana=ventana_giro_f, fps=fps)
    _, vec_j_giro_antes = velocidad_y_direccion(hist_j_giro, frame_min + ventana_giro_f, ventana=ventana_giro_f, fps=fps)
    _, vec_j_giro_desp = velocidad_y_direccion(hist_j_giro, frame_min + ventana_giro_f * 2, ventana=ventana_giro_f, fps=fps)

    angulo_i = angulo_entre(vec_i_giro_antes, vec_i_giro_desp) if (vec_i_giro_antes and vec_i_giro_desp) else 0.0
    angulo_j = angulo_entre(vec_j_giro_antes, vec_j_giro_desp) if (vec_j_giro_antes and vec_j_giro_desp) else 0.0

    vel_max_antes = max(vel_i_antes or 0, vel_j_antes or 0)
    movimiento_real = vel_max_antes > UMBRAL_VEL_ACERCAMIENTO_PXSEG

    return {
        "id_i": id_i, "id_j": id_j,
        "frame": frame_min,
        "acercandose": acercandose,
        "movimiento_real": movimiento_real,
        "dist_min": dist_min,
        "dist_min_ratio": dist_min / tam_prom_en_min if tam_prom_en_min else None,
        "delta_acercamiento": delta_acercamiento,
        "delta_separacion": delta_separacion,
        "separandose": separandose,
        "vel_i_antes": vel_i_antes or 0.0, "vel_j_antes": vel_j_antes or 0.0,
        "vel_i_desp": vel_i_desp or 0.0, "vel_j_desp": vel_j_desp or 0.0,
        "vel_max_antes": vel_max_antes,
        "ratio_frenazo_i": ratio_frenazo_i if ratio_frenazo_i is not None else 1.0,
        "ratio_frenazo_j": ratio_frenazo_j if ratio_frenazo_j is not None else 1.0,
        "ratio_frenazo_min": min(
            ratio_frenazo_i if ratio_frenazo_i is not None else 1.0,
            ratio_frenazo_j if ratio_frenazo_j is not None else 1.0,
        ),
        "angulo_i": angulo_i, "angulo_j": angulo_j,
        "angulo_max": max(angulo_i, angulo_j),
        "n_frames_antes": len(dists_antes),
        "n_frames_despues": len(dists_despues),
    }


def evaluar_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro, fps=FPS_REFERENCIA):
    """
    Mismos gates, mismo sistema de puntos de siempre. A fps=30 (el valor
    por defecto) el comportamiento es identico al original. Para videos
    con otro FPS, ahora se pasa el fps real para que las velocidades y
    ventanas de tiempo esten en las unidades correctas.
    """
    f = calcular_features_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro, fps=fps)
    if f is None:
        return None
    if not (f["acercandose"] and f["movimiento_real"]):
        return None

    frenazo_i = f["ratio_frenazo_i"] < 0.5 and f["vel_i_antes"] > UMBRAL_VEL_FRENAZO_PXSEG
    frenazo_j = f["ratio_frenazo_j"] < 0.5 and f["vel_j_antes"] > UMBRAL_VEL_FRENAZO_PXSEG
    giro_i = f["angulo_i"] > UMBRAL_ANGULO_GIRO
    giro_j = f["angulo_j"] > UMBRAL_ANGULO_GIRO

    score = 0
    if frenazo_i or frenazo_j:
        score += PUNTOS_FRENAZO
    if giro_i or giro_j:
        score += PUNTOS_GIRO
    if f["separandose"]:
        score += PUNTOS_SEPARACION
    if f["vel_max_antes"] > UMBRAL_VEL_ALTA_PXSEG:
        score += PUNTOS_VELOCIDAD_ALTA

    if score < UMBRAL_SCORE:
        return None

    razones = []
    if frenazo_i or frenazo_j:
        razones.append("FRENAZO")
    if giro_i or giro_j:
        razones.append("GIRO")
    if f["separandose"]:
        razones.append("SEPARACION")
    if f["vel_max_antes"] > UMBRAL_VEL_ALTA_PXSEG:
        razones.append("VELOCIDAD_ALTA")

    return {"frame": f["frame"], "par": (id_i, id_j), "razones": razones, "dist_min": f["dist_min"], "score": score}