import base64
import hashlib
import struct

import numpy as np

# Cache global pour stocker les LUTs pré-calculées
# Clé : hash du buffer, Valeur : np.ndarray de forme (1024, 256)
saved_luts: dict[str, np.ndarray] = {}


def get_calibration_fingerprint(calibration_raw_bytes):
    return hashlib.sha256(calibration_raw_bytes).hexdigest()


def generate_texisense_calibration_data(buffer):
    """
    Analyzes the calibration buffer (46,170 bytes).
    Returns a list containing the calibration points (raw_value, pressure_pascal)
    for each sensor (32x32 = 1024).
    """
    CHUNK_SIZE = 1026
    NUM_CHUNKS = 45
    NUM_SENSORS = 1024

    if len(buffer) < CHUNK_SIZE * NUM_CHUNKS:
        raise ValueError("Buffer too small")

    # 1. Buffer Parsing
    pressures = []  # List of pressures (P)
    sensor_data = [[] for _ in range(NUM_SENSORS)]  # Raw data (R) per sensor

    for i in range(NUM_CHUNKS):
        offset = i * CHUNK_SIZE
        # The first 2 bytes are the pressure in Pascal (Unsigned Short)
        p_val = struct.unpack("<H", buffer[offset : offset + 2])[0]
        pressures.append(p_val)

        # The next 1024 bytes are the sensor responses
        # Note: We assume here that sensors are 1 byte (0-255)
        # as suggested by the buffer. If it were 12-bit/2-byte, it would need adjustment.
        raw_values = struct.unpack("1024B", buffer[offset + 2 : offset + 1026])
        for s in range(NUM_SENSORS):
            sensor_data[s].append(raw_values[s])

    # 2. Phase Separation
    # Find the index of the maximum pressure to separate ramp-up and ramp-down phases
    idx_max = pressures.index(max(pressures))

    final_calibration = []

    # 3. Processing per Sensor
    for s in range(NUM_SENSORS):
        # Ascending phase: from start to max
        asc_p = pressures[: idx_max + 1]
        asc_r = sensor_data[s][: idx_max + 1]

        # Descending phase: from max to end
        desc_p = pressures[idx_max:]
        desc_r = sensor_data[s][idx_max:]

        # Create a map to average values by pressure level
        pressure_map = {}

        # Add the essential (0,0) point
        pressure_map[0] = [0]

        # Group raw values by identical pressure levels
        for p, r in zip(asc_p, asc_r):
            if p not in pressure_map:
                pressure_map[p] = []
            pressure_map[p].append(r)

        for p, r in zip(desc_p, desc_r):
            if p not in pressure_map:
                pressure_map[p] = []
            pressure_map[p].append(r)

        # Averaging and sorting by raw value (R)
        sorted_points = []
        for p in sorted(pressure_map.keys()):
            avg_r = sum(pressure_map[p]) / len(pressure_map[p])
            sorted_points.append((avg_r, p))

        # 4. Extrapolation
        # Calculate the slope over the last two steps to avoid the "plateau" effect
        if len(sorted_points) >= 2:
            r2, p2 = sorted_points[-1]
            r1, p1 = sorted_points[-2]
            if r2 != r1:
                slope = (p2 - p1) / (r2 - r1)
                # Add a virtual point far beyond (e.g., max ADC 4095)
                # If your data is 8-bit (0-255), use 255.
                r_extrapol = 255
                p_extrapol = p2 + slope * (r_extrapol - r2)
                sorted_points.append((float(r_extrapol), float(p_extrapol)))

        final_calibration.append(sorted_points)

    return final_calibration


def generate_lut(calibration_data):
    """
    Transforme les points de pivot en une Look-Up Table (LUT) 1024x256.
    On pré-calcule l'interpolation pour chaque valeur brute possible (0-255).
    """
    # On crée une grille de 256 valeurs (0, 1, 2, ..., 255)
    raw_range = np.arange(256)
    lut = np.zeros((1024, 256), dtype=np.float32)

    for s in range(1024):
        points = calibration_data[s]
        xp = [p[0] for p in points]  # Raw
        fp = [p[1] for p in points]  # Pressure

        # Interpolation numpy sur toute la plage 0-255 d'un coup
        # left/right gèrent les cas hors bornes (paliers 0 et max)
        interp_values = np.interp(raw_range, xp, fp)

        # Optionnel : Si tu veux garder ton extrapolation spécifique au-delà du dernier point
        # au lieu de plafonner (comportement par défaut de np.interp)
        last_raw = xp[-1]
        if last_raw < 255 and len(xp) >= 2:
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) if (xp[-1] != xp[-2]) else 0
            # On remplace les valeurs après le dernier point connu par l'extrapolation
            extrapol_mask = raw_range > last_raw
            interp_values[extrapol_mask] = fp[-1] + slope * (
                raw_range[extrapol_mask] - last_raw
            )

        lut[s] = interp_values

    return lut


def apply_texisense_calibration(raw_matrix, calibration_base64):
    """
    Applique la calibration de manière ultra-optimisée en utilisant une LUT.
    """
    calibration_raw = base64.b64decode(calibration_base64)
    h = get_calibration_fingerprint(calibration_raw)

    # 1. Gestion du cache de la LUT
    if h not in saved_luts:
        # On génère les points de pivots (ton ancienne fonction)
        calib_points = generate_texisense_calibration_data(calibration_raw)
        # On transforme ces points en une table de recherche (LUT)
        saved_luts[h] = generate_lut(calib_points)

    lut = saved_luts[h]

    # 2. Application "Magique" (Lookup vectorisé)
    # On aplatit la matrice d'entrée (32x32 -> 1024)
    flat_raw = np.array(raw_matrix, dtype=np.int32).flatten()

    # On crée un index pour chaque capteur (0 à 1023)
    sensor_indices = np.arange(1024)

    # L'indexation NumPy fait le lookup instantanément :
    # Pour chaque capteur 'i', on prend la valeur dans lut[i, valeur_brute_du_capteur_i]
    calibrated_flat = lut[sensor_indices, flat_raw]

    # 3. Retour au format 32x32
    return calibrated_flat.reshape(32, 32).transpose()
