import json
import os

RAW_RECORD_EXTENSION = ".phy"
LEGACY_RAW_RECORD_EXTENSION = ".txt"
RAW_RECORD_EXTENSIONS = (RAW_RECORD_EXTENSION, LEGACY_RAW_RECORD_EXTENSION)
RAW_RECORD_MIME_TYPE = "application/vnd.phyling.record"


def is_raw_record_file(filename: str) -> bool:
    """Return True if the filename has a raw record extension (.phy, or legacy .txt), case-insensitive."""
    return os.path.splitext(filename)[1].lower() in RAW_RECORD_EXTENSIONS


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
