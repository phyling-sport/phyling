import base64
import hashlib
import struct

import numpy as np

CHUNK_SIZE = 1026  # one pressure step: 2 bytes of pressure + one response byte per sensor
NUM_SENSORS = 1024
MAT_SIZE = 32
RAW_LEVELS = 256  # a sensor response is one byte
# one gf/cm2 in Pascal: the calibration steps are stored in gf/cm2, as the vendor software reads them
TEXISENSE_PRESSURE_UNIT_PA = 98.0665
# 3x3 spatial smoothing the vendor software applies to every map, recovered from its per-cell export
SMOOTHING_KERNEL = np.array([[1, 1, 1], [1, 3, 1], [1, 1, 1]], dtype=np.float64)

# TexicarePro model, as the vendor software exports it: the mat metadata report none of it (they
# announce 470 mm / 15 mm steps). Texisense order: row r from the front, column c from the left,
# cell k = r * 32 + c. Sensor s of the mat is Texisense cell (31 - s // 32, s % 32).
TEXICARE_PRO_GEOMETRY = {
    "col_widths_mm": np.array(
        [15.0, *[30.0] * 4, 24.0, *[18.0] * 8, 29.25, 40.5, 35.5, 24.25, *[18.0] * 6]
        + [20.0, *[22.0] * 4, 22.5, 23.0, 11.5]
    ),
    "row_heights_mm": np.array([13.5, 27.0, 26.5, 24.5, *[23.0] * 8, 20.0, *[17.0] * 18, 8.5]),
}

# Cache global pour stocker les LUTs pré-calculées
# Clé : hash du buffer, Valeur : np.ndarray de forme (1024, 256)
saved_luts: dict[str, np.ndarray] = {}


def get_calibration_fingerprint(calibration_raw_bytes, weight_asc=1.0, weight_desc=1.0, threshold_pa=0.0):
    """Build the cache key of a lookup table.

    The phase weights and the pressure threshold shape the table itself, so two mats sharing a
    calibration buffer but not their metadata must not share a table. Orientation is applied after
    the lookup and has nothing to do here.

    Parameters:
        calibration_raw_bytes (bytes): raw calibration blob read from the mat
        weight_asc (float): weight of the ascending phase
        weight_desc (float): weight of the descending phase
        threshold_pa (float): pressure floor, in Pascal

    Returns:
        str: the cache key
    """
    digest = hashlib.sha256(calibration_raw_bytes).hexdigest()
    return f"{digest}:{weight_asc}:{weight_desc}:{threshold_pa}"


def _metadata_number(metadata, key, default):
    """Read one numeric metadata field, falling back on the legacy default when it is absent."""
    value = metadata.get(key, None)
    return default if value is None else float(value)


def read_lut_metadata(metadata):
    """Read the metadata shaping the lookup table: phase weights and resting pressure threshold.

    Every key is optional, metadata block included: mats read before the firmware reported it, and
    every record already stored, carry none of them. Missing weights average the two phases evenly,
    a missing threshold leaves no floor at all - in both cases the historical behaviour.

    Parameters:
        metadata (dict): texisense_metadata block of the mat, possibly empty

    Returns:
        tuple: (weight_asc, weight_desc, threshold_pa) as floats
    """
    weight_asc = _metadata_number(metadata, "weight_asc", 1.0)
    weight_desc = _metadata_number(metadata, "weight_desc", 1.0)
    if weight_asc + weight_desc == 0:
        weight_asc, weight_desc = 1.0, 1.0
    return weight_asc, weight_desc, _metadata_number(metadata, "threshold_pa", 0.0)


def generate_texisense_calibration_data(buffer):
    """Parse the calibration buffer into its pressure steps and per-sensor responses.

    The buffer is a whole number of 1026-byte chunks, each holding one pressure step. The step
    count varies from one mat to another, so it is derived from the buffer length rather than
    assumed: the mat sweeps the pressure up then back down, with a step count that may differ
    between the two phases.

    Each step pressure is a little-endian word in gf/cm2, converted here to Pascal. Reading it
    big-endian in Pascal, as done before, gave pressures 256 / 98.0665 = 2.61 times too high: the
    vendor per-cell export matches the little-endian reading to 0.2 %.

    Parameters:
        buffer (bytes): raw calibration blob read from the mat

    Returns:
        tuple: (pressures, raws) with pressures of shape (n,) in Pascal and raws of shape (n, 1024)
    """
    if len(buffer) < CHUNK_SIZE or len(buffer) % CHUNK_SIZE != 0:
        raise ValueError(f"Calibration buffer of {len(buffer)} bytes is not a multiple of {CHUNK_SIZE}")

    num_steps = len(buffer) // CHUNK_SIZE
    words = [struct.unpack_from("<H", buffer, i * CHUNK_SIZE)[0] for i in range(num_steps)]
    pressures = np.array(words, dtype=np.float64) * TEXISENSE_PRESSURE_UNIT_PA
    raws = np.array([np.frombuffer(buffer, np.uint8, NUM_SENSORS, i * CHUNK_SIZE + 2) for i in range(num_steps)])
    return pressures, raws


def _phase_lut(pressures, raws):
    """Build the raw-to-pressure table of a single calibration phase, for a single sensor.

    Points are averaged per raw value then sorted by raw value, which is what np.interp needs: fed a
    non-monotonic axis it returns silently wrong values. A (0, 0) anchor is added because a mat does
    not necessarily report a zero-pressure step, and a null response is pinned to 0 Pa instead of
    being averaged: a cell answering 0 reports no load detected, whatever the bench applied at that
    step, so a cell still mute over the first steps must not inherit their mean. A sensor that
    saturates answers the same value on several steps, so the averaged curve is forced
    non-decreasing rather than left to dip. Past the last response of the phase np.interp holds
    the last pressure: generate_lut replaces everything above the top-step response anyway.

    Parameters:
        pressures (np.ndarray): pressure of each step of the phase, in Pascal
        raws (np.ndarray): response of that sensor at each step of the phase

    Returns:
        np.ndarray: shape (256,), the pressure in Pascal for every possible raw value
    """
    raw = np.concatenate(([0], raws)).astype("float64")
    press = np.concatenate(([0], pressures)).astype("float64")

    uniq_raw, inverse = np.unique(raw, return_inverse=True)
    uniq_press = np.bincount(inverse, weights=press) / np.bincount(inverse)
    # a cell mute over the first steps averages them into a non-zero floor: no response means no load
    uniq_press[uniq_raw == 0] = 0
    # a saturated sensor answers the same value on several steps: averaging them must not make pressure drop
    uniq_press = np.maximum.accumulate(uniq_press)

    return np.interp(np.arange(RAW_LEVELS), uniq_raw, uniq_press)


def blend_phases(lut_asc, lut_desc, weight_asc, weight_desc):
    """Mix the ascending and descending phase tables into the table used for the lookup.

    This is the one place where the two phases are weighted: the metadata weights only have a
    confirmed meaning for 0/0, read as 1/1, which is an even average - the average the vendor
    per-cell export agrees with.

    Parameters:
        lut_asc (np.ndarray): table of the ascending phase, in Pascal
        lut_desc (np.ndarray): table of the descending phase, in Pascal
        weight_asc (float): weight of the ascending phase
        weight_desc (float): weight of the descending phase

    Returns:
        np.ndarray: the weighted average of the two tables, in Pascal
    """
    return (weight_asc * lut_asc + weight_desc * lut_desc) / (weight_asc + weight_desc)


def generate_lut(pressures, raws, weight_asc=1.0, weight_desc=1.0, threshold_pa=0.0):
    """Build the lookup table turning each sensor's raw response into a pressure in Pascal.

    The ascending and descending phases are interpolated separately and their two tables averaged,
    as prescribed by the manufacturer. Averaging the raw responses instead interleaves the steps of
    the two phases, whose pressure levels do not coincide, and breaks the monotonicity np.interp
    relies on. Above raw_top, the response of a sensor at the highest pressure step P_top, the
    pressure is extended in proportion, P = P_top x raw / raw_top, as the vendor software does:
    prolonging the slope of the last segment tripled the pressure of a cell loaded past its
    calibrated range. That line wins over both phases, even where hysteresis made the descending
    phase answer above raw_top at a lower step: those points are dropped, not blended. The table
    stays increasing without any clamp: at raw_top both phases hold the top step, so their blend
    is at most P_top, below the line from raw_top + 1 on.
    The threshold is baked into the table rather than applied frame by frame: an unloaded cell
    still answers a few LSB, and flooring them here costs nothing at lookup time.

    Parameters:
        pressures (np.ndarray): pressure of each calibration step, in Pascal
        raws (np.ndarray): shape (n, 1024), response of every sensor at each step
        weight_asc (float): weight of the ascending phase in the average
        weight_desc (float): weight of the descending phase in the average
        threshold_pa (float): pressure below which a cell reads 0, in Pascal

    Returns:
        np.ndarray: shape (1024, 256) float32, indexed by sensor then by raw value
    """
    idx_max = int(np.argmax(pressures))
    asc, desc = slice(0, idx_max + 1), slice(idx_max, None)

    lut = np.zeros((NUM_SENSORS, RAW_LEVELS), dtype=np.float32)
    for s in range(NUM_SENSORS):
        lut[s] = blend_phases(
            _phase_lut(pressures[asc], raws[asc, s]),
            _phase_lut(pressures[desc], raws[desc, s]),
            weight_asc,
            weight_desc,
        )

    raw_top = raws[idx_max].astype(np.float64)[:, None]
    raw_range = np.arange(RAW_LEVELS)[None, :]
    beyond = (raw_range > raw_top) & (raw_top > 0)
    proportional = pressures[idx_max] * raw_range / np.maximum(raw_top, 1)
    lut = np.where(beyond, proportional, lut).astype(np.float32)

    if threshold_pa > 0:
        lut[lut < threshold_pa] = 0
    return lut


def _kernel_sum(values):
    """Sum every cell of a 2D map with its 8 neighbours, weighted by SMOOTHING_KERNEL.

    Cells outside the map count as zero. The kernel is symmetric, so correlation and convolution
    coincide and the result does not depend on the orientation of the map.
    """
    rows, cols = values.shape
    padded = np.pad(np.asarray(values, dtype=np.float64), 1)
    total = np.zeros((rows, cols))
    for dr in range(3):
        for dc in range(3):
            total += SMOOTHING_KERNEL[dr, dc] * padded[dr : dr + rows, dc : dc + cols]
    return total


def smooth_pressure_map(pressure_map):
    """Apply the vendor 3x3 smoothing to a calibrated pressure map.

    Each cell becomes the average of itself (weight 3) and its 8 neighbours (weight 1). On the edges
    and corners the sum is divided by the weights that fall inside the mat, not by the full 11, so
    a uniform map stays uniform up to its border.

    Parameters:
        pressure_map (np.ndarray): 2D map of pressures, in Pascal

    Returns:
        np.ndarray: the smoothed map, same shape, float32, in Pascal
    """
    weights = _kernel_sum(np.ones(np.shape(pressure_map)))
    return (_kernel_sum(pressure_map) / weights).astype(np.float32)


def apply_texisense_calibration(raw_matrix, calibration_base64, metadata=None):
    """Turn a 32x32 matrix of raw sensor responses into pressures in Pascal.

    Parameters:
        raw_matrix: 32x32 matrix of raw sensor responses (0-255)
        calibration_base64 (str): calibration blob read from the mat, base64 encoded
        metadata (dict): texisense_metadata block of the mat, absent on every record already stored
            and on every mat not read since the firmware started reporting it

    The chain is the vendor one: lookup table per sensor, resting threshold (baked into the table,
    so applied before smoothing), reordering into Texisense order, then the 3x3 smoothing. The
    vendor software also zeroes the centre column 15 after the smoothing; we deliberately keep it,
    so a load on that column differs from the vendor figures. The lookup is done per sensor,
    before the reordering, which only moves cells of the calibrated map. Realtime and records both
    decode through here, so they show the same map.

    The output follows the vendor export ("1st at front-left, next at right"): out[r, c] with row
    0 at the front of the mat and column 0 on its left, so flattening it row-major gives the
    vendor cell k = r * 32 + c.

    The orientation flags of the metadata are deliberately not applied: the Texisense order is
    recovered from the vendor per-cell export, and the flags' exact meaning is unknown - the
    manufacturer uses them to remap the calibration blob, in code we do not have. They are
    stored, waiting for that answer.

    Returns:
        np.ndarray: shape (32, 32) float32, in Pascal, in Texisense order
    """
    metadata = metadata or {}
    weight_asc, weight_desc, threshold_pa = read_lut_metadata(metadata)
    calibration_raw = base64.b64decode(calibration_base64)
    h = get_calibration_fingerprint(calibration_raw, weight_asc, weight_desc, threshold_pa)

    if h not in saved_luts:
        pressures, raws = generate_texisense_calibration_data(calibration_raw)
        saved_luts[h] = generate_lut(pressures, raws, weight_asc, weight_desc, threshold_pa)

    lut = saved_luts[h]
    flat_raw = np.array(raw_matrix, dtype=np.int32).flatten()
    by_sensor = lut[np.arange(NUM_SENSORS), flat_raw].reshape(MAT_SIZE, MAT_SIZE)
    # sensor rows run from the back of the mat: mirror them into Texisense order
    calibrated = by_sensor[::-1]
    return smooth_pressure_map(calibrated)
