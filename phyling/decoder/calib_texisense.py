import base64
import hashlib
import struct

import numpy as np

CHUNK_SIZE = (
    1026  # one pressure step: 2 bytes of pressure + one response byte per sensor
)
NUM_SENSORS = 1024
MAT_SIZE = 32
RAW_LEVELS = 256  # a sensor response is one byte

# Cache global pour stocker les LUTs pré-calculées
# Clé : hash du buffer, Valeur : np.ndarray de forme (1024, 256)
saved_luts: dict[str, np.ndarray] = {}


def get_calibration_fingerprint(
    calibration_raw_bytes, weight_asc=1.0, weight_desc=1.0, threshold_pa=0.0
):
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

    Parameters:
        buffer (bytes): raw calibration blob read from the mat

    Returns:
        tuple: (pressures, raws) with pressures of shape (n,) in Pascal and raws of shape (n, 1024)
    """
    if len(buffer) < CHUNK_SIZE or len(buffer) % CHUNK_SIZE != 0:
        raise ValueError(
            f"Calibration buffer of {len(buffer)} bytes is not a multiple of {CHUNK_SIZE}"
        )

    num_steps = len(buffer) // CHUNK_SIZE
    # pressure is big-endian: read little-endian the mat caps at 250 Pa, when its own software reports 9797 Pa
    pressures = np.array(
        [struct.unpack_from(">H", buffer, i * CHUNK_SIZE)[0] for i in range(num_steps)]
    )
    raws = np.array(
        [
            np.frombuffer(buffer, np.uint8, NUM_SENSORS, i * CHUNK_SIZE + 2)
            for i in range(num_steps)
        ]
    )
    return pressures, raws


def _phase_lut(pressures, raws):
    """Build the raw-to-pressure table of a single calibration phase, for a single sensor.

    Points are averaged per raw value then sorted by raw value, which is what np.interp needs: fed a
    non-monotonic axis it returns silently wrong values. A (0, 0) anchor is added because a mat does
    not necessarily report a zero-pressure step, and a null response is pinned to 0 Pa instead of
    being averaged: a cell answering 0 reports no load detected, whatever the bench applied at that
    step, so a cell still mute over the first steps must not inherit their mean. Past the highest
    calibrated response the last slope is extended, to avoid the plateau the manufacturer warns
    about - the further from that step, the less accurate the extrapolation. A sensor that saturates
    answers the same value on several steps, so the averaged curve is forced non-decreasing rather
    than left to dip and extrapolate downwards.

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

    raw_range = np.arange(RAW_LEVELS)
    lut = np.interp(raw_range, uniq_raw, uniq_press)

    if (
        len(uniq_raw) >= 2
        and uniq_raw[-1] < RAW_LEVELS - 1
        and uniq_raw[-1] != uniq_raw[-2]
    ):
        slope = (uniq_press[-1] - uniq_press[-2]) / (uniq_raw[-1] - uniq_raw[-2])
        beyond = raw_range > uniq_raw[-1]
        lut[beyond] = uniq_press[-1] + slope * (raw_range[beyond] - uniq_raw[-1])
    return lut


def generate_lut(pressures, raws, weight_asc=1.0, weight_desc=1.0, threshold_pa=0.0):
    """Build the lookup table turning each sensor's raw response into a pressure in Pascal.

    The ascending and descending phases are interpolated separately and their two tables averaged,
    as prescribed by the manufacturer. Averaging the raw responses instead interleaves the steps of
    the two phases, whose pressure levels do not coincide, and breaks the monotonicity np.interp
    relies on. The threshold is baked into the table rather than applied frame by frame: an unloaded
    cell still answers a few LSB, and flooring them here costs nothing at lookup time.

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
    total_weight = weight_asc + weight_desc

    lut = np.zeros((NUM_SENSORS, RAW_LEVELS), dtype=np.float32)
    for s in range(NUM_SENSORS):
        lut[s] = (
            weight_asc * _phase_lut(pressures[asc], raws[asc, s])
            + weight_desc * _phase_lut(pressures[desc], raws[desc, s])
        ) / total_weight

    if threshold_pa > 0:
        lut[lut < threshold_pa] = 0
    return lut


def apply_texisense_calibration(raw_matrix, calibration_base64, metadata=None):
    """Turn a 32x32 matrix of raw sensor responses into pressures in Pascal.

    Parameters:
        raw_matrix: 32x32 matrix of raw sensor responses (0-255)
        calibration_base64 (str): calibration blob read from the mat, base64 encoded
        metadata (dict): texisense_metadata block of the mat, absent on every record already stored
            and on every mat not read since the firmware started reporting it

    The orientation flags of the metadata are deliberately not applied: the mat already came out
    the right way up, and their exact meaning is unknown - the manufacturer uses them to remap the
    calibration blob, in code we do not have. They are stored, waiting for that answer.

    Returns:
        np.ndarray: shape (32, 32) float32, in Pascal
    """
    metadata = metadata or {}
    weight_asc, weight_desc, threshold_pa = read_lut_metadata(metadata)
    calibration_raw = base64.b64decode(calibration_base64)
    h = get_calibration_fingerprint(
        calibration_raw, weight_asc, weight_desc, threshold_pa
    )

    if h not in saved_luts:
        pressures, raws = generate_texisense_calibration_data(calibration_raw)
        saved_luts[h] = generate_lut(
            pressures, raws, weight_asc, weight_desc, threshold_pa
        )

    lut = saved_luts[h]
    flat_raw = np.array(raw_matrix, dtype=np.int32).flatten()
    calibrated_flat = lut[np.arange(NUM_SENSORS), flat_raw]
    return calibrated_flat.reshape(MAT_SIZE, MAT_SIZE).transpose()
