import re
import struct
import logging
import ujson
import time
import datetime
from packaging import version

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
}

cpdef unsigned int getSizeElem(str valType):
    if valType in sizeElemDict:
        return sizeElemDict[valType]
    raise Exception("Invalid type: {}".format(valType))


cpdef str getTypeElem(str valType):
    if valType.startswith("uint"):
        return "uint"
    if valType.startswith("int"):
        return "int"
    if valType.startswith("float"):
        return "float"
    raise Exception("Invalid type: {}, must be uintX, intX or floatX".format(valType))


cpdef object getElem(char * content, int curPos, str valType, int content_size=0):
    cdef int size = getSizeElem(valType)
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


cpdef bint filterValTooHighBeforeCalib(object curMod, object modValNamed, object modVal, str curModName):
    errValTooHIGH = False
    for key, val in modValNamed.items():
        if val == "T":
            continue

        if curMod["type"] == "adc" or curMod["type"] == "analog":
            if modVal[val] < 0 or modVal[val] > 25:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]} (before calibration)")
                return False

        elif curMod["type"] in ("miniphyling", "nanophyling", "ble"):
            if val.startswith("adc_"):
                if modVal[val] < 0 or modVal[val] > 25:
                    logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]} (before calibration)")
                    return False
    return True


cpdef bint filterValTooHighAfterCalib(object curMod, object modValNamed, object modVal, str curModName):
    errValTooHIGH = False
    for key, val in modValNamed.items():
        if val == "T":
            continue
        maxval = 10**10 if val != "gpstimeUs" else 10**16  # year 2286 in us
        if abs(modVal[val]) > maxval:
            logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
            return False

        if val.startswith("acc_"):
            if abs(modVal[val]) > 600.0:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False

        elif val.startswith("gyro_"):
            if abs(modVal[val]) > 5000.0:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False

        elif val.startswith("mag_"):
            if abs(modVal[val]) > 100.0:
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False

        elif curMod["type"] == "polar":
            if val == "HeartBeat" and (modVal[val] < 0 or modVal[val] > 300):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False
            if val == "SensorContact" and (modVal[val] < -1 or modVal[val] > 1):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False

        elif curMod["type"] == "gps":
            if val in ("longitude", "latitude") and (modVal[val] < -200 or modVal[val] > 200):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False
            if val == "speed" and (modVal[val] < 0 or modVal[val] > 1000):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False
            if val == "PDOP" and (modVal[val] < 0 or modVal[val] > 300):
                logSpam.warning(f"Max value reached {curModName}[{val}] = {modVal[val]}")
                return False
    return True


cpdef object loadOne(dict header, char * content, int curPos, dict calib_dict=None, int content_size=0, bint check_higher_values=True):
    cdef str curModName
    cdef object curMod
    cdef double modTime
    cdef object modVal
    cdef object modValNamed
    cdef dict data = None
    cdef int tmpCurPos = curPos

    setup_header(header)  # create all variables if needed
    missingByteSize = 0
    while content_size == 0 or tmpCurPos < content_size:
        tmpCurPos = curPos + missingByteSize
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
            if epochUs / 1e6 < 946684800 or epochUs / 1e6 > 2524608000:  # if epoch is before 2000 or after 2050
                logSpam.warning(f"[Time recalibration] Epoch is not valid ({epochUs / 1e6}s)")
                missingByteSize += 1
                continue
            if header["description"]["epoch"] > 946684800:
                # if recalibration is for more than 7 days, cancel it
                if epochUs / 1e6 < header["description"]["epoch"] - 604800 or epochUs / 1e6 > header["description"]["epoch"] + 604800:
                    logSpam.warning(f"[Time recalibration] Epoch is not valid ({epochUs / 1e6}s), cannot recalibrate more than 7 days")
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
            return newData, size + missingByteSize + TIME_MODULE_SIZE, timeSec

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
            modVal["T"] = (modTime - header["description"]["epochUs"]) / 1000000  # time in seconds since rec start
            if modVal["T"] > 3600 * 24 * 10 or modVal["T"] < -100:  # if time if over 10 days or in the past
                missingByteSize += 1
                continue
            modValNamed["T"] = "T"

            if check_higher_values and not filterValTooHighBeforeCalib(curMod, modValNamed, modVal, curModName):  # filter high values
                missingByteSize += 1
                continue
            if calib_dict is not None and calib_dict != {}:
                modVal = calib.calibration(modVal, curModName, calib_dict)
            if check_higher_values and not filterValTooHighAfterCalib(curMod, modValNamed, modVal, curModName):  # filter high values
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
    if data:
        msg = f"Missing some data ({missingByteSize} bytes at {modVal['T']}s)"
    else:
        msg = f"Missing some data ({missingByteSize} bytes from position {curPos})"
    if missingByteSize > 0:  # if we have lost some data
        logSpam.warning(msg)
        if not data:  # if it's impossible to decode some data
            raise Exception(msg)
    return data, missingByteSize + curMod["size"], modTime / 1000000


cpdef object getCalibration(str filename):
    cdef object calibration = ""
    cdef str type = ""
    cdef bytes ln
    cdef bytes filecontent = S3.get_file_bytes(filename)
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


cpdef object updateCalibration(str filename, str oldFilename, object calibration):
    cdef str type = ""
    cdef bytes ln
    cdef bytes filecontent = S3.get_file_bytes(filename)

    if not S3.file_exists(oldFilename):
        logging.info(f"Save a copy of file in {oldFilename}")
        S3.copy_file(filename, oldFilename)

    pattern = rb"<== calibration ==>\n(.*?)<== data ==>\n"
    replacement = f"<== calibration ==>\n{ujson.dumps(calibration, 4)}\n<== data ==>\n".encode()
    filecontent = re.sub(pattern, replacement, filecontent, flags=re.DOTALL)

    S3.add_file_bytes(filename, filecontent)


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
    cdef dict jsonData = {"modules": {}}
    cdef int statsAll = 0
    cdef dict stats = {}
    for modName in header["modules"].keys():
        stats[modName] = 0
    cdef double lastTime = 0
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
            percent = round(curPos / len(content) * 100)
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

        # if first data saving
        if module_name not in jsonData["modules"]:
            jsonData["modules"][module_name] = {
                "description": {},
                "data": {},
                "data_info": {},
            }
            cols = ["rate", "name"]
            for col in cols:
                if col in header["modules"][module_name]:
                    jsonData["modules"][module_name]["description"][col] = header["modules"][module_name][col]
            jsonData["modules"][module_name]["data"]["T"] = []
            jsonData["modules"][module_name]["data_info"]["T"] = {"unit": "s", "description": ""}
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
                jsonData["modules"][module_name]["data_info"][
                    realVarName
                ] = {"unit": description[i]["unit"], "description": descr}

        # save data
        if timeSec > lastTime:
            lastTime = timeSec
        for name in newData["data"].keys():
            jsonData["modules"][newData["name"]]["data"][name].append(
                newData["data"][name]
            )

        if statsAll % 10000 == 0:
            percent = round(curPos / len(content) * 100)
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
                record.add_error_msg(msg)
            else:
                logging.error(msg)

        # High range gyro processing
        if calibration is not None and mod in calibration and "high_range_gyro" in calibration[mod]:
            try:
                mod_data = calib.high_range_gyro(mod_data, calibration[mod]["high_range_gyro"], record)
            except Exception as e:
                msg = f"Error in high_range_gyro for mod {mod}: {e}"
                if record:
                    record.add_error_msg(msg)
                else:
                    logging.error(msg)

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
        percent = round(curPos / len(content) * 100)
        printDecodingInfos(statsAll, percent, verbose=verbose, record=record)

    dt = datetime.datetime.fromtimestamp(header["description"]["epochUs"] / float(1e6))
    logging.info(f"Record started on {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}. Time precision: {header['description']['timePrecisionUs'] / 1000:.3f}ms")
    logging.info("total: {} data".format(statsAll))
    for key, val in stats.items():
        logging.info("\t{}: {} datas".format(key, val))
    logging.info("<== decode end [{}] ==>".format("SUCCESS" if retSuccess else "ERROR"))
    logging.info("File decoded in {:.3f}s".format(time.time() - start))
    if record:
        record.set_decoding_state("Record successfully decoded")
    if retSuccess:
        return jsonData
    else:
        raise Exception("Error during decoding")
