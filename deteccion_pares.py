import numpy as np

def distancia(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def centro(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def punto_inferior(box):
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
# PARÁMETROS CONGELADOS — 2026-08-29 (sin tocar, se siguen usando
# como GATES de candidatos. Lo que cambia es la decisión final,
# no el filtro de qué pares se consideran candidatos.)
# ============================================================
VENTANA_ANTES = 15
VENTANA_DESPUES = 15
VENTANA_GIRO = 6
UMBRAL_VEL_ACERCAMIENTO = 3
UMBRAL_VEL_ALTA = 6
UMBRAL_ANGULO_GIRO = 40
RATIO_CONTACTO = 0.9

# Sistema de puntuación viejo (se mantiene solo por compatibilidad /
# comparación A-B contra el clasificador nuevo; evaluar_par ya no lo
# necesita para decidir si USAR_CLASIFICADOR=True más abajo)
PUNTOS_FRENAZO = 2
PUNTOS_GIRO = 2
PUNTOS_SEPARACION = 1
PUNTOS_VELOCIDAD_ALTA = 1
UMBRAL_SCORE = 3


def calcular_features_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro):
    """
    Calcula TODAS las magnitudes continuas de un par candidato, sin aplicar
    ningún umbral de decisión final (solo los gates físicos mínimos: hay
    suficientes frames y hay posibilidad de contacto). Pensado para:
      1) ser reusado por evaluar_par (una sola fuente de verdad del cálculo)
      2) alimentar recolectar_features.py, que junta ejemplos de MUCHOS
         videos (incluyendo tráfico pesado sin choque) para entrenar un
         clasificador que reemplace el sistema de puntos.

    Devuelve None si no hay datos suficientes o no hay posibilidad física
    de contacto (mismos 2 chequeos que tenía evaluar_par antes de armar
    el score). Si no, devuelve un dict con features continuas + los
    booleanos de gate (acercandose, movimiento_real) que siguen siendo
    pre-requisito para considerar el par como candidato serio.
    """
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

    # Magnitud de acercamiento/separación (no solo el booleano)
    delta_acercamiento = (dists_antes[0] - dists_antes[-1]) if len(dists_antes) >= 3 else 0.0
    delta_separacion = (dists_despues[-1] - dists_despues[0]) if len(dists_despues) >= 3 else 0.0

    vel_i_antes, _ = velocidad_y_direccion(hist_i, frame_min - 2)
    vel_i_desp, _ = velocidad_y_direccion(hist_i, frame_min + VENTANA_DESPUES)
    vel_j_antes, _ = velocidad_y_direccion(hist_j, frame_min - 2)
    vel_j_desp, _ = velocidad_y_direccion(hist_j, frame_min + VENTANA_DESPUES)

    def ratio_frenazo(v_antes, v_desp):
        # < 1 = frenó, > 1 = aceleró, 1 = igual. None si falta dato.
        if v_antes and v_desp is not None and v_antes > 0:
            return v_desp / v_antes
        return None

    ratio_frenazo_i = ratio_frenazo(vel_i_antes, vel_i_desp)
    ratio_frenazo_j = ratio_frenazo(vel_j_antes, vel_j_desp)

    _, vec_i_giro_antes = velocidad_y_direccion(hist_i_giro, frame_min + VENTANA_GIRO, ventana=VENTANA_GIRO)
    _, vec_i_giro_desp = velocidad_y_direccion(hist_i_giro, frame_min + VENTANA_GIRO * 2, ventana=VENTANA_GIRO)
    _, vec_j_giro_antes = velocidad_y_direccion(hist_j_giro, frame_min + VENTANA_GIRO, ventana=VENTANA_GIRO)
    _, vec_j_giro_desp = velocidad_y_direccion(hist_j_giro, frame_min + VENTANA_GIRO * 2, ventana=VENTANA_GIRO)

    angulo_i = angulo_entre(vec_i_giro_antes, vec_i_giro_desp) if (vec_i_giro_antes and vec_i_giro_desp) else 0.0
    angulo_j = angulo_entre(vec_j_giro_antes, vec_j_giro_desp) if (vec_j_giro_antes and vec_j_giro_desp) else 0.0

    vel_max_antes = max(vel_i_antes or 0, vel_j_antes or 0)
    movimiento_real = vel_max_antes > UMBRAL_VEL_ACERCAMIENTO

    return {
        "id_i": id_i, "id_j": id_j,
        "frame": frame_min,
        # gates (mismos de siempre, se mantienen como pre-filtro)
        "acercandose": acercandose,
        "movimiento_real": movimiento_real,
        # features continuas para el clasificador
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


def evaluar_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro):
    """
    Sin cambios de comportamiento respecto al original: mismos gates,
    mismo sistema de puntos, mismo umbral. Ahora internamente reusa
    calcular_features_par para no duplicar el cálculo.
    """
    f = calcular_features_par(id_i, id_j, serie_distancias, tam_prom_en_min, hist_i, hist_j, hist_i_giro, hist_j_giro)
    if f is None:
        return None
    if not (f["acercandose"] and f["movimiento_real"]):
        return None

    frenazo_i = f["ratio_frenazo_i"] < 0.5 and f["vel_i_antes"] > 4
    frenazo_j = f["ratio_frenazo_j"] < 0.5 and f["vel_j_antes"] > 4
    giro_i = f["angulo_i"] > UMBRAL_ANGULO_GIRO
    giro_j = f["angulo_j"] > UMBRAL_ANGULO_GIRO

    score = 0
    if frenazo_i or frenazo_j:
        score += PUNTOS_FRENAZO
    if giro_i or giro_j:
        score += PUNTOS_GIRO
    if f["separandose"]:
        score += PUNTOS_SEPARACION
    if f["vel_max_antes"] > UMBRAL_VEL_ALTA:
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
    if f["vel_max_antes"] > UMBRAL_VEL_ALTA:
        razones.append("VELOCIDAD_ALTA")

    return {"frame": f["frame"], "par": (id_i, id_j), "razones": razones, "dist_min": f["dist_min"], "score": score}
