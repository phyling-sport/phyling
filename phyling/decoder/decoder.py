import os

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
    filenames = [file for file in os.listdir(path) if (file[-4:] == ".txt")]
    for file in filenames:
        res = decodeSave(path + file, verbose, overwrite)
        if not res:
            print(f"Could not decode file {file} ...")
    return filenames
