import numpy as np

def distancia(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def centro(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def punto_inferior(box):
    # Punto inferior central del bbox: mejor referencia en perspectiva,
    # usado SOLO para el calculo de giro (no para distancia/contacto/frenazo)
    return ((box[0] + box[2]) / 2, box[3])

def tamano_promedio(box):
    return ((box[2]-box[0]) + (box[3]-box[1])) / 2

def velocidad_y_direccion(posiciones_ordenadas, frame_referencia, ventana=8):
    usados = [p for f, p in posiciones_ordenadas if f <= frame_referencia][-ventana:]
    if len(usados) < 2:
        return None, None
    p_ini, p_fin = usados[0], usados[-1]
    df = len(usados) - 1
    vec = ((p_fin[0]-p_ini[0])/df, (p_fin[1]-p_ini[1])/df)
    vel = (vec[0]**2 + vec[1]**2) ** 0.5
    return vel, vec

def angulo_entre(v1, v2):
    m1 = (v1[0]**2 + v1[1]**2) ** 0.5
    m2 = (v2[0]**2 + v2[1]**2) ** 0.5
    if m1 == 0 or m2 == 0:
        return 0
    cos_ang = max(-1, min(1, (v1[0]*v2[0] + v1[1]*v2[1]) / (m1*m2)))
    return np.degrees(np.arccos(cos_ang))

# ============================================================
# PARÁMETROS CONGELADOS — 2026-08-29
# ------------------------------------------------------------
# Estos valores son DEFINITIVOS tras el proceso de ajuste sobre
# el dataset de desarrollo (7 videos, ver tabla de métricas en
# README). NO modificar sin repetir la evaluación completa y
# documentar la razón del cambio + impacto en Precisión/Recall/F1.
#
# Métrica de referencia con estos valores (dev set, 7 videos):
#   TP=3, FN=0, FP=1, TN=4 → Precisión 75%, Recall 100%, F1 86%, FPR 20%
# ============================================================
VENTANA_ANTES = 15
VENTANA_DESPUES = 15
VENTANA_GIRO = 6
UMBRAL_VEL_ACERCAMIENTO = 3
UMBRAL_VEL_ALTA = 6
UMBRAL_ANGULO_GIRO = 40
RATIO_CONTACTO = 0.9

# Sistema de puntuación (congelado junto con lo anterior)
PUNTOS_FRENAZO = 2
PUNTOS_GIRO = 2
PUNTOS_SEPARACION = 1
PUNTOS_VELOCIDAD_ALTA = 1
UMBRAL_SCORE = 3

def evaluar_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro):
    if len(serie_distancias) < 5:
        return None

    dist_min = min(d for f, d in serie_distancias)
    frame_min = [f for f, d in serie_distancias if d == dist_min][0]

    if dist_min >= tam_prom_en_min * RATIO_CONTACTO:
        return None

    dists_antes = [d for f, d in serie_distancias if frame_min - VENTANA_ANTES <= f < frame_min]
    dists_despues = [d for f, d in serie_distancias if frame_min < f <= frame_min + VENTANA_DESPUES]

    acercandose = len(dists_antes) >= 3 and dists_antes[0] > dists_antes[-1] + 5
    separandose = len(dists_despues) >= 3 and dists_despues[-1] > dists_despues[0] + 5

    vel_i_antes, _ = velocidad_y_direccion(hist_i, frame_min - 2)
    vel_i_desp, _ = velocidad_y_direccion(hist_i, frame_min + VENTANA_DESPUES)
    vel_j_antes, _ = velocidad_y_direccion(hist_j, frame_min - 2)
    vel_j_desp, _ = velocidad_y_direccion(hist_j, frame_min + VENTANA_DESPUES)

    frenazo_i = bool(vel_i_antes and vel_i_desp and vel_i_antes > 4 and vel_i_desp < vel_i_antes * 0.5)
    frenazo_j = bool(vel_j_antes and vel_j_desp and vel_j_antes > 4 and vel_j_desp < vel_j_antes * 0.5)

    # Vectores de giro: usan hist_*_giro (bottom-center), NO hist_i/hist_j
    _, vec_i_giro_antes = velocidad_y_direccion(hist_i_giro, frame_min + VENTANA_GIRO, ventana=VENTANA_GIRO)
    _, vec_i_giro_desp = velocidad_y_direccion(hist_i_giro, frame_min + VENTANA_GIRO * 2, ventana=VENTANA_GIRO)
    _, vec_j_giro_antes = velocidad_y_direccion(hist_j_giro, frame_min + VENTANA_GIRO, ventana=VENTANA_GIRO)
    _, vec_j_giro_desp = velocidad_y_direccion(hist_j_giro, frame_min + VENTANA_GIRO * 2, ventana=VENTANA_GIRO)

    giro_i = bool(vec_i_giro_antes and vec_i_giro_desp and angulo_entre(vec_i_giro_antes, vec_i_giro_desp) > UMBRAL_ANGULO_GIRO)
    giro_j = bool(vec_j_giro_antes and vec_j_giro_desp and angulo_entre(vec_j_giro_antes, vec_j_giro_desp) > UMBRAL_ANGULO_GIRO)

    vel_max_antes = max(vel_i_antes or 0, vel_j_antes or 0)
    movimiento_real = vel_max_antes > UMBRAL_VEL_ACERCAMIENTO

    if not (acercandose and movimiento_real):
        return None

    score = 0
    if frenazo_i or frenazo_j:
        score += PUNTOS_FRENAZO
    if giro_i or giro_j:
        score += PUNTOS_GIRO
    if separandose:
        score += PUNTOS_SEPARACION
    if vel_max_antes > UMBRAL_VEL_ALTA:
        score += PUNTOS_VELOCIDAD_ALTA

    if score < UMBRAL_SCORE:
        return None

    razones = []
    if frenazo_i or frenazo_j:
        razones.append("FRENAZO")
    if giro_i or giro_j:
        razones.append("GIRO")
    if separandose:
        razones.append("SEPARACION")
    if vel_max_antes > UMBRAL_VEL_ALTA:
        razones.append("VELOCIDAD_ALTA")

    return {"frame": frame_min, "par": (id_i, id_j), "razones": razones, "dist_min": dist_min, "score": score}