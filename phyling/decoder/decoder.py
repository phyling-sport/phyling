import logging
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import ujson

from phyling.decoder.decoder_utils import *  # noqa
from phyling.decoder.decoder_utils import decode


def decodeSave(filename, verbose=True, overwrite=False):
    """Decode and save decoded data in json format.

    Parameters:
        filename (str): access path + filename of the file to decode.
        verbose (bool): if True, print status infos during decoding.
        overwrite (bool): if False, an already decoded file will not be decoded again.

    Returns:
        Boolean: True if decoding is successful, else False.
    """
    if filename[-4:].lower() != ".txt":
        raise ValueError("Input file is not a txt file")

    fileout = filename[:-4] + ".json"
    if not overwrite and os.path.isfile(fileout):
        print("File already decoded (set overwrite to true to decode)")
        return True

    try:
        jsonData = decode(filename, verbose, use_s3=False)
        print("Write to {}...".format(fileout))
        with open(fileout, "w") as f:
            f.write(ujson.dumps(jsonData))
        return True
    except Exception as e:
        print("Could not decode file, error: ", e)
        return False


def decodeSaveFolder(path, verbose=True, overwrite=False):
    """Decode and save all txt files in a folder.

    Parameters:
        path (str): access path of the folder.
        verbose (bool): if True, print status infos during decoding.
        overwrite (bool): if False, an already decoded file will not be decoded again.
    """
    filenames = [file for file in os.listdir(path) if (file[-4:].lower() == ".txt")]
    for file in filenames:
        res = decodeSave(path + file, verbose, overwrite)
        if not res:
            print(f"Could not decode file {file} ...")
    return filenames


def getTypeGraph(valType: str) -> str:
    """
    Get the type for graphing purposes (number or arrayA or arrayAxB)

        * number, uintX, intX, floatX => number
        * arrayA_<type>, arrayA => arrayA
        * arrayAxB_<type>, arrayAxB => arrayAxB
        * else => exception
    """
    if (
        valType == "number"
        or valType.startswith("uint")
        or valType.startswith("int")
        or valType.startswith("float")
    ):
        return "number"
    if valType.startswith("array"):
        split = valType.split("_")
        if len(split) == 2:
            return split[0]  # arrayA or arrayAxB
        elif len(split) == 1:
            return valType  # arrayA or arrayAxB without type
    raise Exception(
        "Invalid type: {}, must be uintX, intX, floatX, arrayA_<type>, arrayAxB_<type>".format(
            valType
        )
    )


def interp1d_(
    df: pd.DataFrame,
    cols: List[str],
    t_int: Optional[np.ndarray] = None,
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
    cols1: Union[str, List[str]] = "all",
    cols2: Union[str, List[str]] = "all",
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


def fuse(
    res: dict,
    modules: list,
    output_fs: float,
    prefix: bool = False,
    t0: Optional[dict] = None,
    x: str = "T",
) -> pd.DataFrame:
    """Performs fusion of DataFrames at a given sampling rate.

    Parameters:
        res: dict of DataFrames to fuse.
        modules: list of module names to fuse.
        output_fs (float): sampling frequency of the output DataFrame.
        prefix (bool): if True, add module as prefix to column names.
        t0: object with synchronisation times for each module.
        x (str): Time column name.

    Returns:
        Fused DataFrame.
    """
    mod2fuse = [module for module in modules if len(res[module]) > 1]
    if len(mod2fuse) == 0:
        raise ValueError("Sequence does not contain any data.")

    result = res[mod2fuse[0]]
    prefix1 = f"{mod2fuse[0]}." if prefix else ""
    tref = t0[mod2fuse[0]] if t0 is not None else 0
    if len(mod2fuse) > 1:
        for i in range(len(mod2fuse) - 1):
            dt = t0[mod2fuse[i + 1]] - tref if t0 is not None else 0
            prefix2 = f"{mod2fuse[i + 1]}." if prefix else ""
            result = fuse_data(
                result,
                res[mod2fuse[i + 1]],
                "all",
                "all",
                prefix1=prefix1,
                prefix2=prefix2,
                fs=output_fs,
                dt=dt,
                type_="union",
                interp_df1=True,
                x=x,
            )
            prefix1 = ""

    result[x] = result[x] - tref
    result = result.reset_index(drop=True)
    return result


def get_module_rate(data: dict, module: str) -> float:
    """Gets the sampling rate of the input module.

    Parameters:
        data: Object containing all data.
        module (str): name of the module.

    Returns:
        Float with the module sampling rate.
    """
    if module not in data["modules"]:
        raise ValueError(f"Module {module} not found in data.")

    if module == "gps":
        fs = min(data["modules"][module]["description"]["rate"], 10)
    else:
        fs = data["modules"][module]["description"]["rate"]
    return fs


def get_rate(data: dict, modules: Union[str, List[str]]) -> Dict[str, float]:
    """Gets the sampling rates of the input modules.

    Parameters:
        data: Object containing all data.
        modules: str or list of str with the names of the modules for
            which we want to get the sampling rate.

    Returns:
        Float or dict with the sampling rates.
    """
    if isinstance(modules, str):
        modules = [modules]
    elif not isinstance(modules, list):
        raise ValueError("Argument 'modules' should be either string or list.")

    fs = {}
    for module in modules:
        fs[module] = get_module_rate(data, module)
    return fs


def rename_cols(col: str, module: str, sep: str = ".") -> str:
    """Change column name by adding module name.

    Parameters:
        col (str): Column name.
        module (str): Module name.
        sep (str): Separator between module and column name. Defaults to ".".

    Returns:
        New column name, for example: module.col
    """
    if col == "T":
        return col
    else:
        return module + sep + col


def datamodule2df(data: dict, module: str) -> pd.DataFrame:
    """Convert data object from Maxi-Phyling for one module to dataframe.
    Note that the default output data type is "float32" !

    Parameters:
        data (dict): input data from Maxi-Phyling
        module (str): module for which to generate dataframe

    Returns:
        DataFrame
    """
    df = pd.DataFrame()
    if module in data["modules"]:
        if "T" in data["modules"][module]:
            df["T"] = np.array(data["modules"][module]["T"], dtype="float32")
        for field, arr in data["modules"][module]["data"].items():
            dtype = "float32"
            if len(arr) > 0 and isinstance(arr[0], (list, np.ndarray)):
                # Store array in one column
                # df[field] = [np.array(a) for a in arr]
                # Or flatten the array for each time point
                n = len(arr)
                flattened = np.array(arr, dtype=dtype).reshape(n, -1)
                cols = [f"{field}_{i}" for i in range(flattened.shape[1])]
                df = pd.concat([df, pd.DataFrame(flattened, columns=cols)], axis=1)
                continue
            if field == "gpstimeUs":
                dtype = "int64"
            elif field in ("longitude", "latitude"):
                dtype = "float64"
            df[field] = np.array(arr, dtype=dtype)
    return df


def data2df(
    data: dict, modules: Union[str, List[str]] = "all"
) -> Dict[str, pd.DataFrame]:
    """Convert data object from Maxi-Phyling to dataframe(s).

    Parameters:
        data (dict): input data from Maxi-Phyling
        modules (str or list of str): modules from which we generate dataframes

    Returns:
        DataFrame or dict of DataFrame
    """
    if modules == "all":
        modules = list(data["modules"].keys())
    elif not isinstance(modules, list):
        raise ValueError("modules should be a list of str or 'all'")

    res = {}
    for module in modules:
        res[module] = datamodule2df(data, module)
    return res


def json2csv(data: dict, output_fs: Optional[float] = None) -> pd.DataFrame:
    """Process record data.

    Parameters:
        data: Object with input data.
        output_fs (float): sampling frequency of the output DataFrame.
            If None, uses the highest rate among all modules.

    Returns:
        Fused DataFrame with all modules, columns prefixed by module name (e.g. imu.acc_x).
    """
    df = data2df(data, "all")
    modules = list(df.keys())
    fs = get_rate(data, modules)
    if output_fs is None:
        output_fs = np.max(list(fs.values()))

    for mod in modules:
        # Compute sampling frequency for each module
        df[mod]["fs"] = 1 / df[mod]["T"].diff()
        # Rename columns before fusion
        df[mod] = df[mod].rename(lambda col: rename_cols(col, mod), axis="columns")

    # Fuse data
    result = fuse(df, modules, output_fs)  # type: ignore

    # Add absolute datetime column from startingTimeUs
    if "description" in data and "startingTimeUs" in data["description"]:
        t0_us = data["description"]["startingTimeUs"]
        result["time"] = (
            pd.to_datetime(
                (t0_us + result["T"] * 1e6).astype("int64"), unit="us", utc=True
            )
            .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            .str[:-3]
            + "Z"
        )

    return result
