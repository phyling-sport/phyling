import json
import logging

import numpy as np
import pandas as pd


def load_json(filepath):
    """Load a json file.

    Parameters:
        filepath (str): path of the json file.

    Returns:
        Dict.
    """
    f = open(filepath)
    data = json.loads("".join(f.readlines()))
    f.close()
    return data


def deep_merge(d1, d2):
    """Recursively merge two dictionaries."""
    result = d1.copy()
    if d2 is None:
        return result
    for key, value in d2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interp1d_(
    df: pd.DataFrame,
    cols: list[str],
    t_int: None | np.ndarray = None,
    fs: float = 1,
    keep_type: bool = True,
    kind: str = "linear",
    x: str = "T",
) -> pd.DataFrame:
    """Interpolation of df at fixed sampling frequency or using t_int.

    Parameters:
        df (dataFrame): data to interpolate.
        cols (str-array): column names to interpolate.
        t_int: numpy array of time for interpolation. If None, fs is used.
        fs: interpolating sampling frequency.
        keep_type (bool): if True, keep the type of the input data.
        kind (str or int): specifies the kind of interpolation
            ('linear', 'slinear', 'quadratic', 'cubic', ...).
            See scipy interp1d documentation.
        x (str): column name for time.

    Returns:
        Interpolated dataFrame
    """
    from scipy.interpolate import interp1d

    if len(df) < 2:
        raise ValueError("Cannot interpolate: df must have at least 2 entries")

    t = df[x].values
    if t_int is None:
        # Take one less sample to prevent bugs from float representation
        nb_samples = int((t[-1] - t[0]) * fs) - 1
        t_int = np.linspace(0, nb_samples / fs, nb_samples + 1) + t[0]

    f_int = interp1d(t, df[cols], kind=kind, axis=0, bounds_error=False)
    y_int = f_int(t_int)
    interp = pd.DataFrame(data=y_int, columns=cols)
    interp[x] = t_int

    out_cols = [x] + cols
    if keep_type:
        # Apply the original dtype
        dtypes = {k: v for k, v in zip(out_cols, df[out_cols].dtypes)}
        for col in dtypes:
            if dtypes[col] in (np.int64, np.int32):
                # Replace nan values by 0 for int columns
                interp[col] = interp[col].fillna(0)
        interp = interp.astype(dtypes)

    return interp[out_cols]


def fuse_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols1: str | list[str] = "all",
    cols2: str | list[str] = "all",
    prefix1: str = "",
    prefix2: str = "",
    fs: float = 1.0,
    dt: float = 0.0,
    type_: str = "df1",
    interp_df1: bool = False,
    x: str = "T",
    suffix1: str = "",
    suffix2: str = "",
) -> pd.DataFrame:
    """Fusion of 2 DataFrames with different time origin and time shift.

    Parameters:
        df1: first DataFrame to be fused.
        df2: second DataFrame to be fused.
        cols1: list of column names in df1 to keep in the fused DataFrame or "all".
        cols2: list of column names in df2 to keep in the fused DataFrame or "all".
        prefix1 (str): if not empty, prepend it to each col in cols1.
        prefix2 (str): if not empty, prepend it to each col in cols2.
        fs (float): sampling frequency.
        dt (float): time shift between df1 and df2 (if dt>0, df1 is in advance on df2).
        type_ (str): interpolation type, either "df1" (use df1 as reference), "union" or "intersection".
        interp_df1 (bool): if True, both df1 and df2 are interpolated, otherwise only df2 is interpolated.
        x (str): column name for time.
        suffix1 (str): if not empty, append it to each col in cols1.
        suffix2 (str): if not empty, append it to each col in cols2.

    Returns:
        Fused DataFrame.
    """
    if cols1 == "all":
        cols1 = list(df1.columns)
        if "T" in cols1:
            cols1.remove("T")
        if x != "T" and x in cols1:
            cols1.remove(x)
    if cols2 == "all":
        cols2 = list(df2.columns)
        if "T" in cols2:
            cols2.remove("T")
        if x != "T" and x in cols2:
            cols2.remove(x)

    if len(prefix1) > 0 or len(suffix1) > 0:
        mapper = {col: prefix1 + col + suffix1 for col in cols1}
        cols1 = list(mapper.values())
        df1 = df1.rename(columns=mapper)
    if len(prefix2) > 0 or len(suffix2) > 0:
        mapper = {col: prefix2 + col + suffix2 for col in cols2}
        cols2 = list(mapper.values())
        df2 = df2.rename(columns=mapper)

    # Apply time shift
    df2[x] = df2[x] - dt

    # Compute time range
    t1, t2 = df1[x], df2[x]
    if type_ == "df1":
        t_min, t_max = t1.min(), t1.max()
    elif type_ == "intersection":
        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())
        if t_min > t_max:
            raise ValueError("Time ranges do not overlap.")
    elif type_ == "union":
        if max(t1.min(), t2.min()) > min(t1.max(), t2.max()):
            # Concatenate df1 and df2 without interpolation
            logging.info(
                "Time ranges do not overlap. Concatenation without interpolation."
            )
            return pd.concat([df1, df2], ignore_index=True)

        t_min = min(t1.min(), t2.min())
        t_max = max(t1.max(), t2.max())
    else:
        raise ValueError(f"Interpolation type {type_} is not supported.")

    # Interpolate df1 / df2
    if interp_df1:
        T = t_max - t_min
        nb_samples = int(T * fs) - 1
        t_int = np.linspace(0, nb_samples / fs, nb_samples + 1) + t_min
        interp1 = interp1d_(df1, cols1, t_int, keep_type=True, x=x)
        interp2 = interp1d_(df2, cols2, t_int, keep_type=True, x=x)
    else:
        ind1 = (df1[x] >= t_min) & (df1[x] <= t_max)
        interp1 = df1[ind1].reset_index(drop=True)
        interp2 = interp1d_(df2, cols2, interp1[x].values, keep_type=True, x=x)

    df = pd.concat([interp1[cols1], interp2], axis=1, join="inner")
    cols_order = [x] + cols1 + cols2
    return df[cols_order]
