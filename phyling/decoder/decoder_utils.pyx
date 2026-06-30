# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import re
import os
import struct
import logging
import ujson
import time
import datetime
from packaging import version
import shutil

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import phyling.decoder.calibration_use as calib

TIME_MODULE_ID = 100
TIME_MODULE_NAME = "__TIME_UPDATE__"
TIME_MODULE_SIZE = 13

HEADER_UPDATE_DICT = "__header_update__"

try:
    from phylingUtils.data_layer.s3 import S3
except Exception:
    class S3:
        def get_filestream_readonly(filename):
            return open(filename, "rb+")

        @classmethod
        def get_file_bytes(
            cls,
            filename,
        ) -> bytes:
            with open(filename, "rb") as f:
                filecontent = f.read()
            return filecontent


def get_file_bytes_local(
    filename,
) -> bytes:
    with open(filename, "rb") as f:
        filecontent = f.read()
    return filecontent


try:
    from phylingUtils.utils.logging_setup import logSpam
except Exception:
    class logSpam(object):
        @classmethod
        def info(cls, *args, **kwargs):
            logging.info(*args, **kwargs)

        @classmethod
        def warning(cls, *args, **kwargs):
            logging.warning(*args, **kwargs)

        @classmethod
        def error(cls, *args, **kwargs):
            logging.error(*args, **kwargs)

        @classmethod
        def end(cls):
            pass

        @classmethod
        def update(self):
            pass


class EndOfFileException(Exception):
    pass

cdef dict sizeElemDict = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float32": 4,
    "float64": 8,
    # array32_float32
    # array32x32_int16
}

# Pre-compiled regex patterns for array type parsing (avoid recompilation on every call)
_RE_ARRAY_2D = re.compile(r"array(\d+)x(\d+)_(\w+)")
_RE_ARRAY_1D = re.compile(r"array(\d+)_(\w+)")


cpdef dict getArrayInfo(str valType):
    cdef dict info = {
        "dim1": 0,
        "dim2": 0,
        "elemType": "",
        "elemSize": 0,
        "totalSize": 0,
    }

    if valType.startswith("array"):
        match = _RE_ARRAY_2D.match(valType)
        if match:
            info["dim1"] = int(match.group(1))
            info["dim2"] = int(match.group(2))
            info["elemType"] = match.group(3)
            info["elemSize"] = getSizeElem(info["elemType"])
            info["totalSize"] = info["dim1"] * info["dim2"] * info["elemSize"]
            return info
        match = _RE_ARRAY_1D.match(valType)
        if match:
            info["dim1"] = int(match.group(1))
            info["dim2"] = 0
            info["elemType"] = match.group(2)
            info["elemSize"] = getSizeElem(info["elemType"])
            info["totalSize"] = info["dim2"] * info["elemSize"]
            return info
    raise Exception("Invalid type: {}, must be arrayA_<type> or arrayAxB_<type>".format(valType))

cpdef unsigned int getSizeElem(str valType):
    cdef dict arrayInfo

    if valType in sizeElemDict:
        return sizeElemDict[valType]
    if valType.startswith("array"):
        return getArrayInfo(valType)["totalSize"]
    raise Exception("Invalid type: {}".format(valType))


cpdef str getTypeElem(str valType):
    if valType.startswith("uint"):
        return "uint"
    if valType.startswith("int"):
        return "int"
    if valType.startswith("float"):
        return "float"
    if valType.startswith("array"):
        return "array"
    raise Exception("Invalid type: {}, must be uintX, intX, floatX, arrayA_<type>, arrayAxB_<type>".format(valType))


cpdef object getElem(char * content, int curPos, str valType, int content_size=0):
    cdef int size = getSizeElem(valType)

    # For array types
    cdef dict arrayInfo
    cdef list array = []
    cdef list row = []
    cdef int x, y
    cdef int offset = 0

    if content_size > 0 and curPos + size > content_size:
        raise Exception("File is broken, unpack requires a buffer of {} bytes".format(size))
    if getTypeElem(valType) == "uint":
        return int.from_bytes(
            content[curPos : curPos + size],  # noqa
            byteorder="little",
            signed=False,
        )
    if getTypeElem(valType) == "int":
        return int.from_bytes(
            content[curPos : curPos + size],  # noqa
            byteorder="little",
            signed=True,
        )
    if getTypeElem(valType) == "float":
        return float(
            struct.unpack(
                "<f" if size == 4 else "<d", content[curPos : curPos + size]  # noqa
            )[0]
        )
    if getTypeElem(valType) == "array":
        arrayInfo = getArrayInfo(valType)
        for x in range(arrayInfo["dim1"]):
            if arrayInfo["dim2"] == 0:
                array.append(getElem(content, curPos + offset, arrayInfo["elemType"], content_size))
                offset += arrayInfo["elemSize"]
            else:
                row = []
                for y in range(arrayInfo["dim2"]):
                    row.append(getElem(content, curPos + offset, arrayInfo["elemType"], content_size))
                    offset += arrayInfo["elemSize"]
                array.append(row)
        return array


cpdef object applyFactor(object value, object curMod, object elem):
    if curMod["type"] in ("imu", "mag", "miniphyling", "nanophyling", "ble") and elem["type"] == "int16":
        if "acc_factor" in curMod and elem["name"].startswith("acc_"):
            return value * curMod["acc_factor"]
        elif "gyro_factor" in curMod and elem["name"].startswith("gyro_"):
            return value * curMod["gyro_factor"]
        elif "mag_factor" in curMod and elem["name"].startswith("mag_"):
            return value * curMod["mag_factor"]
        elif "adc_factor" in curMod and elem["name"].startswith("adc_"):
            return value * curMod["adc_factor"]
        elif "temp_factor" in curMod and elem["name"].startswith("temp"):  # temp or temperature
            return value * curMod["temp_factor"]
    # elif curMod["type"] in ("adc", "analog") and elem["type"] == "uint16":
    #     if "factor" in curMod:
    #         return value * curMod["factor"]
    elif elem["type"] == "uint16" or elem["type"] == "int16":
        if "factor" in curMod:
            return value * curMod["factor"]
    return value


cpdef str getModName(dict header, char * content, int curPos):
    cdef str modName
    if content[curPos] == TIME_MODULE_ID:
        if version.parse(header["description"]["version"]) >= version.parse("v6.6.0"):
            return TIME_MODULE_NAME
    for modName, mod in header["modules"].items():
        if mod["id"] == content[curPos]:
            return modName
    return ""


cpdef str getVarName(dict header, str modName, str varBaseName):
    if "variablesNames" in header["description"] \
    and header["description"]["variablesNames"] is not None \
    and modName in header["description"]["variablesNames"] \
    and varBaseName in header["description"]["variablesNames"][modName]:
        return header["description"]["variablesNames"][modName][varBaseName]
    return varBaseName


cpdef void setup_header(dict header):
    if "__setup__" in header:  # already setup
        return
    if "deviceType" not in header["description"]:
        header["description"]["deviceType"] = "maxiphyling"
    if "version" not in header["description"]:
        header["description"]["version"] = "v6.0.0"
    if "deviceId" not in header["description"]:
        header["description"]["deviceId"] = -1
    if "epochUs" not in header["description"]:
        header["description"]["epochUs"] = header["description"]["epoch"] * 1e6
    if "epoch" not in header["description"]:
        header["description"]["epoch"] = int(header["description"]["epochUs"] / 1e6)
    if "timePrecisionUs" not in header["description"]:
        header["description"]["timePrecisionUs"] = 1e9  # default precision (no time)
    header["__setup__"] = True


DEF FILTER_KEEP = 1      # data is valid, keep it
DEF FILTER_SKIP = 0      # expected/normal skip (e.g. no GPS fix), not an error
DEF FILTER_ERROR = -1    # value out of valid range, likely corrupted

DEF GPS_OK = 0              # GPS values usable (or no GPS fields) — keep frame unchanged
DEF GPS_SANITIZED = 1       # combined module: invalid GPS fields set to NaN in place (other data kept)
DEF GPS_DROP_NOFIX = 2      # standalone GPS module: no fix (expected) — whole frame dropped silently
DEF GPS_DROP_CORRUPTED = 3  # standalone GPS module: out-of-range values — whole frame dropped, counts as error


cpdef int filterValTooHighBeforeCalib(object curMod, object modValNamed, object modVal, str curModName):
    """Filter decoded module values before calibration.

    Returns FILTER_KEEP (1), FILTER_SKIP (0), or FILTER_ERROR (-1).
    """
    for key, val in modValNamed.items():
        if val in ("T", "epoch"):
            continue
        if not isinstance(modVal[val], (int, float)):
            continue

        if curMod["type"] == "adc" or curMod["type"] == "analog":
            if modVal[val] < 0 or modVal[val] > 25:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]} (before calibration)")
                return FILTER_ERROR

        elif curMod["type"] in ("miniphyling", "nanophyling", "ble"):
            if val.startswith("adc_"):
                if modVal[val] < 0 or modVal[val] > 25:
                    logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]} (before calibration)")
                    return FILTER_ERROR
    return FILTER_KEEP


cpdef int filterValTooHighAfterCalib(object curMod, object modValNamed, object modVal, str curModName):
    """Filter decoded module values after calibration.

    Returns FILTER_KEEP (1), FILTER_SKIP (0), or FILTER_ERROR (-1).
    GPS validity (no fix, out-of-range) is handled separately in sanitizeGpsFields, which
    invalidates the GPS fields to NaN instead of dropping the frame (frames may carry valid IMU).
    """
    for key, val in modValNamed.items():
        if val in ("T", "epoch"):
            continue
        if not isinstance(modVal[val], (int, float)):
            continue

        maxval = 10**10 if "time" not in val else 10**16  # year 2286 in us
        if abs(modVal[val]) > maxval:
            logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
            return FILTER_ERROR

        if val.startswith("acc_"):
            if abs(modVal[val]) > 600.0:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return FILTER_ERROR

        elif val.startswith("gyro_"):
            if abs(modVal[val]) > 5000.0:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return FILTER_ERROR

        elif val.startswith("mag_"):
            if abs(modVal[val]) > 100.0:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return FILTER_ERROR

        elif curMod["type"] == "polar":
            if val == "HeartBeat" and (modVal[val] < 0 or modVal[val] > 300):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return FILTER_ERROR
            if val == "SensorContact" and (modVal[val] < -1 or modVal[val] > 1):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return FILTER_ERROR

    # GPS fields are validated/sanitized to NaN in sanitizeGpsFields (handles both standalone
    # "gps" modules with bare names and "gps_"-prefixed fields of combined miniphyling frames),
    # so they are intentionally not range-checked / dropped here.
    return FILTER_KEEP


cpdef int sanitizeGpsFields(object curMod, object modVal, str curModName):
    """Decide what to do with the GPS fields of a frame when they are not usable.

    Works for both naming schemes:
      - standalone "gps" modules (Maxi GpsModule, raw Mini file) use bare names ("latitude", ...).
        The whole frame is GPS-only, so there is nothing else to keep -> the frame is dropped.
        Returns GPS_DROP_NOFIX (expected no-fix) or GPS_DROP_CORRUPTED (out-of-range values).
      - combined modules (miniphyling / nanophyling / ble) bundle IMU + GPS in one frame with
        "gps_"-prefixed names. Dropping the frame would discard the valid high-frequency IMU data,
        so we invalidate ONLY the GPS position/quality fields in place (-> NaN, returns GPS_SANITIZED)
        while zeroing nSat (signals no-fix to downstream consumers).

    NaN is the platform-wide "missing GPS" representation (interpolation produces a gap instead of
    a teleport, and the map renderer keeps NaN). Modifies modVal in place in the combined case.

    Two invalidity cases:
      - no-fix (nSat == 0, or lat == lon == 0): expected, silent.
      - out-of-range / corrupted values: warned.

    Returns GPS_OK (valid / no GPS), GPS_SANITIZED (combined, NaN'd),
    GPS_DROP_NOFIX (standalone, no fix), or GPS_DROP_CORRUPTED (standalone, corrupted).
    """
    cdef double nan = float("nan")
    cdef str prefix
    if curMod["type"] == "gps":
        prefix = ""
    elif curMod["type"] in ("miniphyling", "nanophyling", "ble"):
        prefix = "gps_"
    else:
        return GPS_OK

    if prefix + "latitude" not in modVal or prefix + "longitude" not in modVal:
        return GPS_OK

    lat = modVal[prefix + "latitude"]
    lon = modVal[prefix + "longitude"]
    nsat = modVal.get(prefix + "nSat", None)
    speed = modVal.get(prefix + "speed", None)
    pdop = modVal.get(prefix + "PDOP", None)

    # NaN inputs (e.g. already sanitized in a prior pass) are treated as no-fix.
    if lat != lat or lon != lon:
        no_fix = True
        corrupted = False
    else:
        no_fix = (nsat is not None and nsat == 0) or (lat == 0 and lon == 0)
        corrupted = (
            lat < -90 or lat > 90
            or lon < -180 or lon > 180
            or (nsat is not None and nsat > 64)
            or (speed is not None and (speed < 0 or speed > 1000))
            or (pdop is not None and (pdop < 0 or pdop > 300))
        )
    if not no_fix and not corrupted:
        return GPS_OK

    if corrupted:
        bad_fields = []
        if lat < -90 or lat > 90:
            bad_fields.append(f"lat={lat}")
        if lon < -180 or lon > 180:
            bad_fields.append(f"lon={lon}")
        if nsat is not None and nsat > 64:
            bad_fields.append(f"nSat={nsat}")
        if speed is not None and (speed < 0 or speed > 1000):
            bad_fields.append(f"speed={speed}")
        if pdop is not None and (pdop < 0 or pdop > 300):
            bad_fields.append(f"PDOP={pdop}")
        logSpam.warning(f"Invalid GPS value in {curModName}: {', '.join(bad_fields)}")

    # Standalone GPS-only frame: nothing else to keep -> let the caller drop it.
    if prefix == "":
        return GPS_DROP_CORRUPTED if corrupted else GPS_DROP_NOFIX

    # Combined frame: keep the IMU data, invalidate only the GPS fields.
    for suffix in ("longitude", "latitude", "speed", "altitude", "PDOP", "heading"):
        if prefix + suffix in modVal:
            modVal[prefix + suffix] = nan
    if prefix + "nSat" in modVal:
        modVal[prefix + "nSat"] = 0
    return GPS_SANITIZED


cpdef object loadOne(dict header, char * content, int curPos, dict calib_dict=None, int content_size=0, bint check_higher_values=True):
    """Decode a single frame at curPos and return (data, size, timestamp).

    Each decoded frame carries two time columns: "T" and "epoch" (both in seconds).
    "epoch" is always the absolute epoch time (modTime). "T" is relative to the record start,
    except when header["description"]["epochUs"] == 0 (stream-outside-record sentinel): there is
    no frozen record epoch, so T == epoch and the 10-day/past sanity clip is skipped.
    """
    cdef str curModName
    cdef object curMod
    cdef double modTime
    cdef object modVal
    cdef object modValNamed
    cdef dict data = None
    cdef int tmpCurPos = curPos

    setup_header(header)  # create all variables if needed
    missingByteSize = 0   # corrupted/unreadable bytes — triggers warning
    skippedByteSize = 0   # expected skips (e.g. GPS no fix) — no warning
    while content_size == 0 or tmpCurPos < content_size:
        tmpCurPos = curPos + missingByteSize + skippedByteSize
        curModName = getModName(header, content, tmpCurPos)
        if curModName == "":
            if content_size == 0:
                raise Exception("module id {} does not exist or content is empty".format(content[curPos]))
            missingByteSize += 1
            continue

        # process time recalibration
        if curModName == TIME_MODULE_NAME:
            precisionUs = getElem(content, tmpCurPos + 1, "uint32", content_size=content_size)
            epochUs = getElem(content, tmpCurPos + 5, "uint64", content_size=content_size)
            if epochUs / 1e6 < 1420070400 or epochUs / 1e6 > 2524608000:  # if epoch is before 2015 or after 2050
                logSpam.warning(f"[Time recalibration] Epoch is not valid ({epochUs / 1e6}s)")
                missingByteSize += 1
                continue
            if not HEADER_UPDATE_DICT in header:
                header[HEADER_UPDATE_DICT] = {}
            header[HEADER_UPDATE_DICT]["epochUs"] = epochUs
            header[HEADER_UPDATE_DICT]["epoch"] = epochUs / 1e6
            header[HEADER_UPDATE_DICT]["timePrecisionUs"] = precisionUs
            dt = datetime.datetime.fromtimestamp(epochUs / float(1e6))
            logging.info(f"[+] Time recalibrated with a precision of {precisionUs / 1000:.3f}ms ({dt.strftime('%Y-%m-%d %H:%M:%S.%f')})")

            if missingByteSize > 0:  # if we have lost some data
                msg = f"Missing some data ({missingByteSize} bytes from position {curPos}) (Before time calibration module)"
                logSpam.warning(msg)
            newData, size, timeSec = loadOne(header, content, tmpCurPos + TIME_MODULE_SIZE, calib_dict, content_size)
            return newData, size + missingByteSize + skippedByteSize + TIME_MODULE_SIZE, timeSec

        curMod = header['modules'][curModName]
        if content_size > 0 and tmpCurPos + curMod["size"] > content_size:
            tmpModName = getModName(header, content, curPos)
            if tmpModName is not "":
                tmpModName = f" lost module is {tmpModName}"
            raise EndOfFileException(f"End of file is corrupted (total: {missingByteSize} bytes corrupted from position {curPos}){tmpModName}")

        try:
            modTime = 0
            modVal = {}
            modValNamed = {}
            for elem in curMod["description"]:
                if elem["name"] == "id":
                    pass
                elif elem["name"] == "time":
                    modTime = getElem(content, tmpCurPos, elem["type"], content_size=content_size)
                else:
                    varName = getVarName(header, curModName, elem["name"])
                    modVal[elem["name"]] = getElem(content, tmpCurPos, elem["type"], content_size=content_size)
                    modVal[elem["name"]] = applyFactor(modVal[elem["name"]], curMod, elem)
                    modValNamed[varName] = elem["name"]
                tmpCurPos += getSizeElem(elem["type"])
            epochUs = header["description"]["epochUs"]
            if epochUs == 0:  # stream outside record sentinel: no frozen epoch, T == absolute epoch, no clip
                modVal["T"] = modTime / 1e6
            else:
                modVal["T"] = (modTime - epochUs) / 1e6  # time in seconds since rec start
                if modVal["T"] > 3600 * 24 * 10 or modVal["T"] < -100:  # if time if over 10 days or in the past
                    missingByteSize += 1
                    continue
            modValNamed["T"] = "T"
            modVal["epoch"] = modTime / 1e6  # absolute epoch time in seconds
            modValNamed["epoch"] = "epoch"

            if check_higher_values:
                beforeCalibResult = filterValTooHighBeforeCalib(curMod, modValNamed, modVal, curModName)
                if beforeCalibResult == FILTER_SKIP:
                    skippedByteSize += curMod["size"]
                    continue
                elif beforeCalibResult == FILTER_ERROR:
                    missingByteSize += 1
                    continue
            if calib_dict is not None and calib_dict != {}:
                modVal = calib.calibration(modVal, curModName, calib_dict)
            if check_higher_values:
                afterCalibResult = filterValTooHighAfterCalib(curMod, modValNamed, modVal, curModName)
                if afterCalibResult == FILTER_SKIP:
                    skippedByteSize += curMod["size"]
                    continue
                elif afterCalibResult == FILTER_ERROR:
                    missingByteSize += 1
                    continue
                # Standalone "gps" frames (GPS-only) are dropped on invalid GPS; combined frames
                # (e.g. miniphyling) keep their IMU and only NaN their GPS fields.
                gps_result = sanitizeGpsFields(curMod, modVal, curModName)
                if gps_result == GPS_DROP_NOFIX:
                    skippedByteSize += curMod["size"]
                    continue
                elif gps_result == GPS_DROP_CORRUPTED:
                    missingByteSize += 1
                    continue

            for key, val in modValNamed.items():
                modValNamed[key] = modVal[val]
            data = {
                "type": curMod["type"],
                "name": curModName,
                "data": modValNamed,
            }
            break
        except Exception as e:
            if content_size == 0:
                raise Exception(f"Error on decoding: {str(e)}")
            logSpam.warning(f"Error on decoding: {str(e)}. trying next module")
            missingByteSize += 1
    if missingByteSize > 0:
        if data:
            msg = f"Missing some data ({missingByteSize} bytes at {modVal.get('T', '?')}s)"
        else:
            msg = f"Missing some data ({missingByteSize} bytes from position {curPos})"
        logSpam.warning(msg)
        if not data:
            raise Exception(msg)
    return data, missingByteSize + skippedByteSize + curMod["size"], modTime / 1e6


cdef object _round_value(object val, int decimals):
    """Round a scalar float or every element of a list/nested list to the given number of decimals.

    Non-float numerics (e.g. numpy int64) are converted to float before rounding to ensure
    JSON serializability, matching the behaviour of utils.roundAll.
    """
    cdef object item
    if isinstance(val, float):
        return round(val, decimals)
    if isinstance(val, list):
        return [_round_value(item, decimals) for item in val]
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return val


cpdef list loadAll(dict header, bytes content, int curPos=5, dict calib_dict=None,
                   bint check_higher_values=True, bint round_values=False, int round_decimals=6):
    """Decode all frames from an MQTT payload in a single C-level loop.

    Iterates over the full payload and calls loadOne for each frame. Each frame's data carries
    both the relative "T" column and the absolute "epoch" column (see loadOne); this realtime
    path keeps "epoch" (unlike the offline decode output).
    If round_values is True, numeric values in data["data"] are rounded before being
    appended to the result list (T is rounded to 3 decimals, all others to round_decimals).
    Handles scalar floats, lists, and nested lists.

    Args:
        header: Record description dict (recDescription).
        content: Raw MQTT payload bytes.
        curPos: Starting byte offset (default 5 = SOCKET_HEADER_SIZE).
        calib_dict: Calibration dict, or None/empty to skip calibration.
        check_higher_values: If True, filter out out-of-range values.
        round_values: If True, round numeric values before appending to result.
        round_decimals: Number of decimal places for rounding (default 6). T is always
                        rounded to 3 decimals regardless of this value.

    Returns:
        List of (data, size, timestamp) tuples for each successfully decoded frame.
    """
    cdef int len_content = len(content)
    cdef list results = []
    cdef object data
    cdef int size
    cdef double timestamp
    cdef str elem_key
    cdef object elem_val

    while curPos < len_content:
        try:
            data, size, timestamp = loadOne(
                header=header,
                content=content,
                curPos=curPos,
                calib_dict=calib_dict,
                check_higher_values=check_higher_values,
            )
        except EndOfFileException:
            # Corrupted end-of-file frame: stop iteration gracefully
            break
        except Exception:
            # Corrupted frame: skip one byte and try to re-sync
            curPos += 1
            continue

        curPos += size

        if round_values:
            for elem_key in list(data["data"].keys()):
                elem_val = data["data"][elem_key]
                data["data"][elem_key] = _round_value(
                    elem_val, 3 if elem_key == "T" else round_decimals
                )

        results.append((data, size, timestamp))

    return results


cpdef object getCalibration(str filename, use_s3=True):
    cdef object calibration = ""
    cdef str type = ""
    cdef bytes ln
    cdef bytes filecontent
    if use_s3:
        filecontent = S3.get_file_bytes(filename)
    else:
        filecontent = get_file_bytes_local(filename)
    for ln in filecontent.splitlines(True):
        if ln == b"":
            break
        if ln == b"<== description ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== calibration ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== data ==>\n":
            type = ln.decode("utf-8")
        else:
            if type == "<== description ==>\n":
                pass
            elif type == "<== calibration ==>\n":
                calibration += ln.decode("utf-8")
            elif type == "<== data ==>\n":
                break
    if len(calibration) > 0:
        calibration = ujson.loads(calibration)
    return calibration


cpdef object updateCalibration(str filename, str oldFilename, object calibration, use_s3=True):
    cdef str type = ""
    cdef bytes ln
    cdef bytes filecontent
    if use_s3:
        filecontent = S3.get_file_bytes(filename, use_s3=use_s3)

        if not S3.file_exists(oldFilename, use_s3=use_s3):
            logging.info(f"Save a copy of file in {oldFilename}")
            S3.copy_file(filename, oldFilename, use_s3=use_s3)

        pattern = rb"<== calibration ==>\n(.*?)<== data ==>\n"
        replacement = f"<== calibration ==>\n{ujson.dumps(calibration, 4)}\n<== data ==>\n".encode()
        filecontent = re.sub(pattern, replacement, filecontent, flags=re.DOTALL)

        S3.add_file_bytes(filename, filecontent, use_s3=use_s3)

    else:
        filecontent = get_file_bytes_local(filename)

        if not os.path.exists(oldFilename):
            logging.info(f"Save a copy of file in {oldFilename}")

            shutil.copyfile(filename, oldFilename)

        pattern = rb"<== calibration ==>\n(.*?)<== data ==>\n"
        replacement = f"<== calibration ==>\n{ujson.dumps(calibration, 4)}\n<== data ==>\n".encode()
        filecontent = re.sub(pattern, replacement, filecontent, flags=re.DOTALL)

        with open(filename, "wb") as f:
            f.write(filecontent)


cpdef object loadFile(str filename, bint verbose=False, double startingTime=-1, bint use_s3=True, object record=None):
    if startingTime == -1:
        startingTime = time.time()
    logging.info("load {}...".format(filename))
    cdef str header = ""
    cdef object calibration = ""
    cdef list content = []
    cdef str type = ""
    cdef int totalSz = 0
    cdef int lastPrintSz = 0
    cdef bytes ln
    cdef bytes content_byte
    cdef object header_dict
    cdef bytes filecontent
    if use_s3:
        filecontent = S3.get_file_bytes(filename)
    else:
        filecontent = get_file_bytes_local(filename)
    cdef bint isDataSection = False
    for ln in filecontent.splitlines(True):
        totalSz += len(ln)
        # print every 10Mb
        if verbose:
            if totalSz - lastPrintSz > 10000000:
                logging.info(
                    "read {}Mb in {:.2f}s".format(
                        int(totalSz / 1000000), time.time() - startingTime
                    )
                )
                lastPrintSz = totalSz - (totalSz % 10000000)
        if ln == b"":
            break
        if isDataSection:
            if ln.endswith(b" ==>\n"):  # there is another description, calibration or data part...
                msg = "File is corrupted with a second description part, stopping reading"
                if record:
                    record.add_error_msg(msg)
                else:
                    logging.error(msg)
                break
            content.append(ln)
        elif ln == b"<== description ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== calibration ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== data ==>\n":
            type = ln.decode("utf-8")
            isDataSection = True
        else:
            if type == "<== description ==>\n":
                header += ln.decode("utf-8")
            elif type == "<== calibration ==>\n":
                calibration += ln.decode("utf-8")
            elif type == "<== data ==>\n":
                content.append(ln)
    # print on the end
    if verbose:
        logging.info(
            "read {:.3f}Mb in {:.2f}s".format(
                totalSz / 1000000, time.time() - startingTime
            )
        )

    content_byte = b"".join(content)
    header_dict = ujson.loads(header)
    if len(calibration) > 0:
        calibration = ujson.loads(calibration)
    else:
        calibration = {}
    return header_dict, calibration, content_byte


cpdef void printDecodingInfos(int statsAll, int percent, bint verbose=True, object record=None):
    if statsAll < 1000:
        msg = "[{percent:3}%]: {val:.0f} data decoded".format(
            percent=percent, val=float(statsAll)
        )
    elif statsAll < 1000000:
        msg = "[{percent:3}%]: {val:.0f}k data decoded".format(
            percent=percent, val=float(statsAll / 1000)
        )
    else:
        msg = "[{percent:3}%]: {val:.2f}M data decoded".format(
            percent=percent, val=float(statsAll / 1000000.0)
        )
    if verbose:
        logging.info(msg)
    if record:
        record.set_decoding_state(f"{msg}")


cpdef dict decode(str filename, bint verbose=True, dict config_client=None, object record=None, bint use_s3=True, bint check_higher_values=True):
    """Decode a record file into a jsonData dict of modules.

    The per-frame "epoch" column produced by loadOne is realtime-only and is excluded from this
    offline output (only "T" and the regular data columns are kept).
    """
    logging.info("<== decode start [{}] ==>".format(filename))
    cdef bint retSuccess = True
    cdef double start = time.time()

    cdef object header
    cdef object calibration
    cdef bytes content
    if record:
        record.set_decoding_state("Loading file")
    header, calibration, content = loadFile(
        filename, verbose=verbose, startingTime=start, use_s3=use_s3, record=record,
    )
    setup_header(header)
    logging.info("start decoding file")
    cdef int curPos = 0
    cdef dict jsonData = {
        "modules": {},
        "description": {},
        "calib": calibration,
    }
    cdef int statsAll = 0
    cdef dict stats = {}
    for modName in header["modules"].keys():
        stats[modName] = 0
    cdef double lastTime = 0
    cdef bint isMini = header["description"]["deviceType"] == "miniphyling"
    cdef dict last_gps_data = {}  # last saved GPS values per module (keyed by module_name)
    cdef int dev_id = header["description"]["deviceId"]

    cdef int percent
    cdef object newData
    cdef int size
    cdef double timeSec
    cdef str module_name
    cdef list description
    cdef int content_size = len(content)
    if version.parse(header["description"]["version"]) <= version.parse("v6.0.0"):
        logging.info(f"{header['description']['deviceType']} old version")
    else:
        logging.info(f"{header['description']['deviceType']} #{header['description']['deviceId']} {header['description']['version']}")
        dt = datetime.datetime.fromtimestamp(header["description"]["epochUs"] / float(1e6))
        logging.info(f"Time starting precision: {header['description']['timePrecisionUs'] / 1000:.3f}ms ({dt.strftime('%Y-%m-%d %H:%M:%S.%f')})")
    if dev_id < 0 and header["description"]["deviceType"] == "maxiphyling":
        try:
            dev_id = int(header["description"]["folder_name"].split("_")[0])
        except Exception:
            dev_id = -1
    elif dev_id < 0 and header["description"]["deviceType"] == "miniphyling":
        try:  # M42_1.TXT
            dev_id = int(header["description"]["folder_name"].split("_")[0][1:])
        except Exception:
            dev_id = -1
    if content_size == 0:
        logging.error("File is empty")
        raise Exception("File is empty")
    while 1:
        if content_size <= curPos:
            break
        if content[curPos] == 0:  # id for stopping parsing
            percent = round(<double>curPos / content_size * 100)
            if percent > 95:
                break
            logSpam.warning("Current module ID is 0, skipping")
        try:
            newData, size, timeSec = loadOne(header, content, curPos, calib_dict=calibration, content_size=content_size, check_higher_values=check_higher_values)
        except EndOfFileException as e:
            logSpam.warning(f"[ERROR]: unexpected error at end of file {e}")
            retSuccess = True
            break
        except Exception as e:
            logSpam.error(f"[ERROR]: unexpected error, {e}")
            # retSuccess = False
            break
        statsAll += 1
        module_name = newData["name"]
        stats[module_name] += 1
        curPos += size

        # GPS deduplication: skip if no GPS values changed since last point
        if newData["type"] == "gps" and isMini:
            new_gps_vals = {}
            for k, v in newData["data"].items():
                if k != "T" and k != "epoch":
                    new_gps_vals[k] = v
            if module_name in last_gps_data and new_gps_vals == last_gps_data[module_name]:
                continue
            last_gps_data[module_name] = new_gps_vals
        # if first data saving
        if module_name not in jsonData["modules"]:
            jsonData["modules"][module_name] = {
                "description": {},
                "data": {},
                "data_info": {},
            }
            cols = ["rate", "type", "name", "bleName"]  # cols to copy in description
            for col in cols:
                if col in header["modules"][module_name]:
                    jsonData["modules"][module_name]["description"][col] = header["modules"][module_name][col]
            jsonData["modules"][module_name]["data"]["T"] = []
            jsonData["modules"][module_name]["data_info"]["T"] = {"unit": "s", "description": "", "type": "number"}
            description = header["modules"][module_name]["description"]
            for i in range(2, len(description)):
                realVarName = getVarName(header, module_name, description[i]["name"])
                descr = ""
                if config_client and module_name in config_client:
                    if description[i]["name"] in config_client[module_name]:
                        descr = config_client[module_name][realVarName][
                            "description"
                        ]
                jsonData["modules"][module_name]["data"][realVarName] = []

                jsonData["modules"][module_name]["data_info"][realVarName] = {
                    "description": descr
                }
                cols = ["unit", "type", "min", "max"]  # cols to copy in data_info (for each data column)
                for col in cols:
                    if col in description[i]:
                        jsonData["modules"][module_name]["data_info"][realVarName][col] = description[i][col]

        # save data
        if timeSec > lastTime:
            lastTime = timeSec
        for name in newData["data"].keys():
            if name == "epoch":  # epoch is realtime-only, excluded from offline output
                continue
            jsonData["modules"][newData["name"]]["data"][name].append(
                newData["data"][name]
            )

        if statsAll % 10000 == 0:
            percent = round(<double>curPos / content_size * 100)
            printDecodingInfos(statsAll, percent, verbose=verbose, record=record)

        logSpam.update()
    logSpam.end()

    for mod in jsonData["modules"].keys():
        mod_data = jsonData["modules"][mod]
        # apply time correction for miniphyling
        try:
            mod_data = calib.mini_processing(mod_data, mod)
        except Exception as e:
            msg = f"Error in mini_processing for mod {mod}: {e}"
            if record:
                record.add_warn_msg(msg)
            else:
                logging.warning(msg)

        # High range gyro processing
        if calibration is not None and mod in calibration and "high_range_gyro" in calibration[mod]:
            try:
                mod_data = calib.high_range_gyro(mod_data, calibration[mod]["high_range_gyro"], record)
            except Exception as e:
                msg = f"Error in high_range_gyro for mod {mod}: {e}"
                if record:
                    record.add_warn_msg(msg)
                else:
                    logging.warning(msg)

    # update some parameters in description (like epoch or time precision)
    if HEADER_UPDATE_DICT in header:
        for key, updated in header[HEADER_UPDATE_DICT].items():
            header["description"][key] = updated

    jsonData["description"] = {
        "nbData": statsAll,
        "totalTime": lastTime - (header["description"]["epochUs"] * 1e-6),
        "startingTime": (header["description"]["epochUs"] * 1e-6),
        "startingTimeUs": header["description"]["epochUs"],
        "timePrecisionUs": header["description"]["timePrecisionUs"],
        "device_id": dev_id,
        "deviceType": header["description"]["deviceType"],
        "version": header["description"]["version"],
        "specificData": {},
        "TZ": "",
    }
    if "specificData" in header["description"] and header["description"]["specificData"] != "":
        try:
            jsonData["description"]["specificData"] = header["description"]["specificData"]
        except Exception:
            jsonData["description"]["specificData"] = {}
            logging.warning("Unable to load specific data")
    if "TZ" in header["description"]:
        jsonData["description"]["TZ"] = header["description"]["TZ"]

    if verbose:
        percent = round(<double>curPos / content_size * 100)
        printDecodingInfos(statsAll, percent, verbose=verbose, record=record)

    dt = datetime.datetime.fromtimestamp(header["description"]["epochUs"] / float(1e6))
    logging.info(f"Record started on {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}. Time precision: {header['description']['timePrecisionUs'] / 1000:.3f}ms")
    logging.info("total: {} data".format(statsAll))
    for key, val in stats.items():
        logging.info("  {}: {} datas".format(key, val))
    logging.info("<== decode end [{}] ==>".format("SUCCESS" if retSuccess else "ERROR"))
    logging.info("File decoded in {:.3f}s".format(time.time() - start))
    if record:
        record.set_decoding_state("Record successfully decoded")
    if retSuccess:
        return jsonData
    else:
        raise Exception("Error during decoding")
