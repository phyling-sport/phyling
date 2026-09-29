import copy
import struct
from array import array

import numpy as np
import ujson

from phyling.decoder.decoder_utils import TIME_MODULE_ID
from phyling.decoder.realtime_dedup import build_frame_sizes
from phyling.decoder.realtime_dedup import SOCKET_HEADER_SIZE

_LEN_STRUCT = struct.Struct("<I")
_TS_STRUCT = struct.Struct("<q")


def pack_payload(payload: bytes) -> bytes:
    """Return a payload prefixed with its u32 length, the unit stored in a realtime record stream and part."""
    return _LEN_STRUCT.pack(len(payload)) + payload


def iter_payloads(blob: bytes):
    """Yield the payloads of a blob of length prefixed payloads (see pack_payload); a truncated tail is ignored.

    Args:
        blob (bytes): Concatenation of pack_payload outputs
    """
    pos = 0
    n = len(blob)
    unpack_len = _LEN_STRUCT.unpack_from
    while pos + 4 <= n:
        size = unpack_len(blob, pos)[0]
        pos += 4
        if pos + size > n:
            return
        yield blob[pos : pos + size]
        pos += size


def build_description(header: dict) -> dict:
    """Return the recDescription written in a rebuilt data.txt: each module rate is its effective realtime rate.

    A realtime record holds the stream at its realtimeRate (decimated when lower than the saving rate): the decoder
    and the analyses read the module "rate" as the sampling frequency, so it must describe the rebuilt data.

    Args:
        header (dict): The recDescription of the record

    Returns:
        dict: A modified deep copy of the recDescription
    """
    description = copy.deepcopy(header)
    for mod in description["modules"].values():
        realtime_rate = mod.get("realtimeRate") or 0
        if 0 < realtime_rate < (mod.get("rate") or 0):
            mod["rate"] = realtime_rate
    return description


def assemble_frames(header: dict, blobs) -> tuple:
    """Split realtime payloads into frames, drop exact duplicates and sort them by time.

    The dedup key is `(module_id, ts)`, without window: within one record no two distinct frames share it. The
    first copy of a frame is kept; frames are then ordered by timestamp (module id, then arrival, on ties). A
    remainder that cannot be split (unknown module id, truncated frame) is dropped, as a time update frame (id 100,
    never streamed) would be: neither can be placed in time.

    Args:
        header (dict): The recDescription of the record
        blobs: Iterable of blobs of length prefixed payloads (see pack_payload)

    Returns:
        tuple: (frames bytes, stats dict with "payloads", "frames", "duplicates", "unsplit" and "time_frames")
    """
    sizes = build_frame_sizes(header)
    sizes_get = sizes.get
    unpack_ts = _TS_STRUCT.unpack_from
    data = bytearray()
    # typed arrays: ~19 B per frame instead of ~150 B with Python lists (hours-long records)
    starts, lengths, mod_ids, timestamps = (
        array("q"),
        array("H"),
        array("B"),
        array("q"),
    )
    stats = {
        "payloads": 0,
        "frames": 0,
        "duplicates": 0,
        "unsplit": 0,
        "time_frames": 0,
    }
    for blob in blobs:
        for payload in iter_payloads(blob):
            stats["payloads"] += 1
            pos = SOCKET_HEADER_SIZE
            n = len(payload)
            while pos < n:
                mod_id = payload[pos]
                size = sizes_get(mod_id)
                if size is None or pos + size > n:
                    stats["unsplit"] += 1
                    break
                if mod_id == TIME_MODULE_ID:
                    stats["time_frames"] += 1
                else:
                    starts.append(len(data))
                    lengths.append(size)
                    mod_ids.append(mod_id)
                    timestamps.append(unpack_ts(payload, pos + 1)[0])
                    data += payload[pos : pos + size]
                pos += size
    if not starts:
        return b"", stats
    ts_arr = np.frombuffer(timestamps, dtype=np.int64)
    mod_arr = np.frombuffer(mod_ids, dtype=np.uint8)
    order = np.lexsort((np.arange(len(ts_arr)), mod_arr, ts_arr))
    ts_sorted = ts_arr[order]
    mod_sorted = mod_arr[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (ts_sorted[1:] != ts_sorted[:-1]) | (mod_sorted[1:] != mod_sorted[:-1])
    kept = order[first]
    stats["duplicates"] = int(len(order) - len(kept))
    stats["frames"] = int(len(kept))
    view = memoryview(data)
    frames = b"".join([view[starts[i] : starts[i] + lengths[i]] for i in kept.tolist()])
    return frames, stats


def build_data_txt(header: dict, calibration: dict, frames: bytes) -> bytes:
    """Build a data.txt in the SD card format from a recDescription, a calibration and sorted frames.

    Args:
        header (dict): The recDescription of the record
        calibration (dict): The device calibration at record start
        frames (bytes): The frames, as returned by assemble_frames

    Returns:
        bytes: The file content
    """
    return b"".join(
        [
            b"<== description ==>\n",
            ujson.dumps(build_description(header)).encode(),
            b"\n<== calibration ==>\n",
            ujson.dumps(calibration or {}).encode(),
            b"\n<== data ==>\n",
            frames,
        ]
    )
