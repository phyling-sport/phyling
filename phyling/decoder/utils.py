import json

import numpy as np
import pandas as pd


def data2df(data, modules):
    """Convert data object from Maxi-Phyling to dataframe(s).

    Parameters:
        data (dict): input data from Maxi-Phyling.
        modules (str or list of str): modules from which we generate dataframes

    Returns:
        DataFrame or dict of DataFrame
    """
    if isinstance(modules, list):
        res = {}
        for module in modules:
            if module in data["modules"]:
                df = pd.DataFrame()
                if "T" in data["modules"][module]:
                    df["T"] = np.array(data["modules"][module]["T"], dtype="float32")
                for field, arr in data["modules"][module]["data"].items():
                    df[field] = np.array(arr, dtype="float32")
                res[module] = df
        return res
    elif modules == "all":
        res = {}
        for module in data["modules"]:
            df = pd.DataFrame()
            if "T" in data["modules"][module]:
                df["T"] = np.array(data["modules"][module]["T"], dtype="float32")
            for field, arr in data["modules"][module]["data"].items():
                df[field] = np.array(arr, dtype="float32")
            res[module] = df
        return res
    else:
        df = pd.DataFrame()
        if "T" in data["modules"][modules]:
            df["T"] = np.array(data["modules"][modules]["T"], dtype="float32")
        for field, arr in data["modules"][modules]["data"].items():
            df[field] = np.array(arr, dtype="float32")
        return df


def df2data(df, cols, data={}, module="processed", data_info={}):
    """Convert dataframe into data object for Maxi-Phyling.

    Parameters:
        df (DataFrame): input data.
        cols: list of str with the columns to keep in the output.
        data (dict): initial data object
        module (str): name of the module in which we add new data.
        data_info: dict {col: {'unit': 'g', 'description': 'Acc'}, ...}
            containing unit and description for each column.

    Returns:
        Dict data object for Maxi-Phyling.
    """
    if not data:
        data["modules"] = {}
    if module in data["modules"].keys():
        field = list(data["modules"][module]["data"].keys())[0]
        if len(df) == len(data["modules"][module]["data"][field]):
            if "data_info" not in data["modules"][module]:
                data["modules"][module]["data_info"] = {}
            for col in cols:
                data["modules"][module]["data"][col] = list(df[col].values)
                if data_info:
                    data["modules"][module]["data_info"][col] = data_info[col]
        else:
            raise ValueError("Length of df does not match length of module.")
    else:
        data["modules"][module] = {}
        data["modules"][module]["data"] = {}
        data["modules"][module]["data_info"] = {}
        data["modules"][module]["data"]["T"] = list(df["T"].values)
        data["modules"][module]["data_info"]["T"] = {"description": "", "unit": "s"}
        for col in cols:
            data["modules"][module]["data"][col] = list(df[col].values)
            if data_info:
                data["modules"][module]["data_info"][col] = data_info[col]

    return data


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
