import logging

import numpy as np


def calibration_1D(data, coef=None, offset=None):
    """
    Parameters:
        data (float): input data
        coef (float): calibration coefficient
        offset (float): calibration offset
    Return:
        calibrated data (float)
    """
    if coef is None:
        coef = 1
    if offset is None:
        offset = 0
    return coef * (data + offset)


def calibration_3D(data, cols, coef=None, offset=None):
    """
    Parameters:
        data (dict): input data
        cols (list): list of columns to calibrate
        coef (3x3 np.matrix): calibration coefficient
        offset (3x1 np.matrix): offset calibration
    """
    if len(cols) != 3:
        raise ValueError("cols should be a list of 3 elements.")

    for col in cols:
        if col not in data:
            raise ValueError(f"Column {col} not in data.")

    if coef is None:
        coef = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if offset is None:
        offset = np.array([0, 0, 0])

    coef = np.array(coef)
    offset = np.array(offset)
    # Check if coef and offset have the right shape
    if coef.shape != (3, 3):
        raise ValueError(f"coef shape should be (3, 3) but is {coef.shape}.")
    if offset.shape != (3,):
        raise ValueError(f"offset shape should be (3,) but is {offset.shape}.")

    arr = np.array([data[col] for col in cols])
    arr = (coef @ (arr + offset).T).T
    for i, col in enumerate(cols):
        data[col] = arr[i]

    return data


def calibration(data, module, calib):
    """Calibrate the data from a module

    Parameters:
        data (dict): input data
        module (str): module name
        calib (dict): calibration coefficients and offsets

    Return:
        data (dict): calibrated data
    """
    if module not in calib:
        return data

    mapper = {
        "acc": ["acc_x", "acc_y", "acc_z"],
        "gyro": ["gyro_x", "gyro_y", "gyro_z"],
        "adc": [],
    }
    for key, value in calib[module].items():
        if key == "high_range_gyro":
            continue
        coef = value["coef"] if "coef" in value else None
        offset = value["offset"] if "offset" in value else None
        if key in mapper:
            if key == "adc":
                # We have 2 possible sets of column names for adc
                cols1 = ["adc_0", "adc_1", "adc_2"]
                cols2 = ["0", "1", "2"]
                if all([col in data for col in cols1]):
                    mapper[key] = cols1
                elif all([col in data for col in cols2]):
                    mapper[key] = cols2
            data = calibration_3D(data, mapper[key], coef, offset)
        elif key in data:
            data[key] = calibration_1D(data[key], coef, offset)

    return data


def r2_score(y_true, y_pred):
    """R^2 (coefficient of determination) regression score function.
    Based on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html.

    Parameters:
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns:
        z (float): R^2 score(s).

    References:
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred do not have the same shapes.")

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if len(y_pred) < 2:
        logging.warning("R^2 score is not well-defined with less than two samples.")
        return float("nan")

    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float32)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(
        axis=0, dtype=np.float32
    )
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    return np.average(output_scores)


def solve_lin_reg(x, y, intercept=True):
    """Determines coefficient of linear regression and intercept between x and y.

    Parameters:
        x: numpy array or list.
        y: numpy array or list.
        intercept (boolean): if True, linear regression with intercept, else 0 intercept.

    Returns:
        Tuple with regression coefficients.
        m: linear coefficient.
        c: intercept.
        r2: regression coefficient.
    """
    x = np.array(x)
    y = np.array(y)
    if np.isnan(x).any():
        y = y[~np.isnan(x)]
        x = x[~np.isnan(x)]
    if np.isnan(y).any():
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
    if intercept:
        A = np.vstack([x, np.ones(len(x))]).T
        p = np.linalg.lstsq(A, y, rcond=None)[0]
        m, c = p[0], p[1]
    else:
        A = np.reshape(x, (-1, 1))
        p = np.linalg.lstsq(A, y, rcond=None)[0]
        m, c = p[0], 0

    r2 = r2_score(y, m * x + c)

    return m, c, r2


def compute_time(T, fs):
    """Transform times from Mini-Phyling (constant time values within packets).

    Parameters:
        T: Mini-Phyling bleTime data array.
        fs (float): sampling frequency.

    Returns:
        Transformed Mini-Phyling time.
    """
    dt = 1 / fs
    N = len(T)
    Tint = np.zeros(N)
    cur_t = T[0]
    k = 0
    Tint[0] = cur_t
    for i in range(1, N):
        if T[i] == cur_t:
            # We are within a packet (constant T)
            k += 1
        else:
            # This is beginning of a new packet
            cur_t = T[i]
            k = 0
        Tint[i] = cur_t + k * dt
    return Tint


def time_correction(bleTime, notifTime, fs):
    """Compute corrected time from Mini-Phyling time & Maxi-Phyling notification time.

    Parameters:
        bleTime: Mini-Phyling time array.
        notifTime: Maxi-Phyling notification time array.
        fs (float): sampling frequency.

    Returns:
        Corrected data time.
    """
    t_ble = compute_time(bleTime, fs)
    m, c, r2 = solve_lin_reg(bleTime, notifTime, intercept=True)
    logging.info(
        f"Time correction v1 with linear regression t_corr = {m}t + {c} (R^2 = {r2})"
    )
    if r2 < 0.999:
        logging.error("Time regression is not good enough, no correction applied ...")
        return t_ble

    t_ble_corr = m * t_ble + c
    return t_ble_corr


def time_correction_v1(mod_data: dict):
    """Compute time correction for MiniPhyling v1.x"""
    fs = mod_data["description"]["rate"]

    T = np.array(mod_data["data"]["T"])
    bleTime = np.array(mod_data["data"]["bleTime"])
    notifTime = np.array(mod_data["data"]["maxiNotifTime"])

    # Preprocess times
    bleTime = (bleTime - bleTime[0]) / 1e3 + T[0]
    notifTime = (notifTime - notifTime[0]) / 1e3 + T[0]

    # Separate data into chunks based on lost data and sensor switch off events
    ind = np.where((np.diff(notifTime) > 0.5) | (np.diff(bleTime) < 0))[0] + 1
    ind = [0] + list(ind) + [len(T)]

    # Compute corrected time for each chunk
    T_corr = []
    for i in range(len(ind) - 1):
        T_corr.extend(
            time_correction(
                bleTime[ind[i] : ind[i + 1]], notifTime[ind[i] : ind[i + 1]], fs
            )
        )
    return np.array(T_corr) - 0.03


def time_correction_v2(mod_data: dict, mod: str):
    """Compute time correction since v6.6.2"""
    MIN_TIME_DIFF_SEC = 15
    MAX_SHIFT_SEC = 1
    T = np.array(mod_data["data"]["T"])
    notifDiff = np.array(mod_data["data"]["notifDiff"])
    # Remove notifDiff outliers
    notifDiff[notifDiff < -MAX_SHIFT_SEC * 1e6] = max(notifDiff)

    # Separate data into chunks based on lost data and sensor switch off events
    ind = np.where(np.diff(T) > 3)[0] + 1
    ind = [0, *list(ind), len(T)]

    # Preprocess times
    T_corr = []
    for i in range(len(ind) - 1):
        duration = T[ind[i + 1] - 1] - T[ind[i]]
        appliedDiff = 0
        # apply correction only if duration is long enough
        if duration > MIN_TIME_DIFF_SEC:
            min_ = np.min(notifDiff[ind[i] : ind[i + 1]])
            max_ = np.max(notifDiff[ind[i] : ind[i + 1]])
            if min_ == 0 and max_ == 0:
                logging.warning(
                    f"{mod}: Cannot apply correction to old miniphyling data"
                )
            else:
                appliedDiff = min_ / 1e6 - 0.003
                # 0.003 is the mean time btw notif sent on mini and receive on maxi
                logging.info(
                    f"{mod}: Applied time offset: {appliedDiff * 1000:4.1f}ms "
                    f"from T {int(T[ind[i]])}s to {int(T[ind[i + 1]-1])}s"
                )
        else:
            logging.warning(
                f"{mod}: Mini connected only {duration:4.1f}s no time offset applied "
                f"(minimum {MIN_TIME_DIFF_SEC}s required)"
            )

        if abs(appliedDiff) > MAX_SHIFT_SEC:
            logging.warning(
                f"{mod}: Time offset {appliedDiff * 1000:4.1f}ms is too large "
                f"(> {MAX_SHIFT_SEC * 1000:4.1f}ms), no correction applied."
            )
            appliedDiff = 0
        T_corr.extend(T[ind[i] : ind[i + 1]] + appliedDiff)
    return T_corr


def mini_processing(mod_data: dict, mod: str):
    """Correct time for MiniPhyling data for better synchronisation."""
    vars = mod_data["data"].keys()
    if "notifDiff" in vars:
        mod_data["data"]["T"] = time_correction_v2(mod_data, mod)
    elif "maxiNotifTime" in vars and "bleTime" in vars:
        mod_data["data"]["T"] = time_correction_v1(mod_data)
    return mod_data


def high_range_gyro(mod_data, mod_calib, record=None):
    """Apply high range gyro calibration to input data."""
    from phylingAnalysis.motiontracking.high_range_gyro import process_high_range

    for key in ["col_gyro", "col_acc", "coef"]:
        if key not in mod_calib:
            raise ValueError(f"Missing key {key} in calibration data.")

    col_gyro = mod_calib["col_gyro"]
    col_acc = mod_calib["col_acc"]
    m = mod_calib["coef"]
    fs = mod_data["description"]["rate"]

    for key in [col_gyro, col_acc]:
        if key not in mod_data["data"]:
            raise ValueError(f"Missing key {key} in data.")

    gyro = np.array(mod_data["data"][col_gyro])
    acc = np.array(mod_data["data"][col_acc])
    mod_data["data"][col_gyro], _, _ = process_high_range(
        gyro, acc, fs, m, record=record
    )
    return mod_data
