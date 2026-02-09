import base64
import hashlib
import struct

import numpy as np

# Cache for previously processed calibration data to avoid redundant computations
saved_calibration_data: dict[str, dict] = {}


def get_calibration_fingerprint(calibration_data):
    """
    Generates a unique fingerprint for the given calibration data using SHA-256.
    """
    return hashlib.sha256(calibration_data).hexdigest()


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


def apply_texisense_calibration(
    raw_matrix, calibration_data=None, calibration_base64=None, use_cache=True
):
    """
    Applies the calibration to a 32x32 raw matrix.

    Args:
        raw_matrix: 32x32 list[list[int]] (raw values 0-255).
        calibration_data: List of 1024 lists, where each sub-list contains
                         the pivot points (raw, pressure) for a specific sensor.
        calibration_base64: Base64 string containing the calibration data.
        use_cache: Whether to use cached calibration data if available.
    Returns:
        np.ndarray: 32x32 matrix of calculated pressures in Pascal.
    """
    if calibration_data is None and calibration_base64 is not None:
        calibration_raw = base64.b64decode(calibration_base64)
        if use_cache:
            hash = get_calibration_fingerprint(calibration_raw)
            if hash not in saved_calibration_data:
                saved_calibration_data[hash] = generate_texisense_calibration_data(
                    calibration_raw
                )
            calibration_data = saved_calibration_data[hash]
        else:
            calibration_data = generate_texisense_calibration_data(calibration_raw)
    elif calibration_data is None:
        raise ValueError(
            "Either calibration_data or calibration_base64 must be provided."
        )

    # Convert to numpy array for easier manipulation (flattening)
    flat_raw = np.array(raw_matrix).flatten()
    if len(flat_raw) != 1024:
        raise ValueError("The input matrix must contain exactly 1024 elements.")

    calibrated_flat = np.zeros(1024)

    for s in range(1024):
        r = flat_raw[s]
        points = calibration_data[s]  # List of tuples (raw, pressure)

        # Extract coordinates for interpolation
        xp = [p[0] for p in points]  # Raw values (X)
        fp = [p[1] for p in points]  # Pressures (Y)

        # Case 1: Raw value below the first point (often 0)
        if r <= xp[0]:
            calibrated_flat[s] = fp[0]

        # Case 2: Raw value above the last step (Extrapolation)
        elif r >= xp[-1]:
            # Calculate the slope between the last two points for extrapolation
            if len(xp) >= 2:
                r_last = xp[-1]
                r_prev = xp[-2]
                p_last = fp[-1]
                p_prev = fp[-2]

                if r_last != r_prev:
                    slope = (p_last - p_prev) / (r_last - r_prev)
                    calibrated_flat[s] = p_last + slope * (r - r_last)
                else:
                    calibrated_flat[s] = p_last
            else:
                calibrated_flat[s] = fp[-1]

        # Case 3: Standard linear interpolation between two steps
        else:
            calibrated_flat[s] = np.interp(r, xp, fp)

    # Reconstruct the 32x32 matrix
    return calibrated_flat.reshape(
        32, 32
    ).transpose()  # Transpose to match original orientation if needed
