import os
import random
import struct
import tempfile
import unittest

from phyling.decoder.decoder_utils import decode
from phyling.decoder.decoder_utils import loadFile
from phyling.decoder.realtime_record import assemble_frames
from phyling.decoder.realtime_record import build_data_txt
from phyling.decoder.realtime_record import build_description
from phyling.decoder.realtime_record import iter_payloads
from phyling.decoder.realtime_record import pack_payload

EPOCH_US = 1_790_000_000_000_000
IMU_ID, GPS_ID = 1, 2
IMU_SIZE, GPS_SIZE = 23, 33


def make_header() -> dict:
    """Build a recDescription with a decimated IMU and a GPS module."""
    return {
        "description": {"version": "v7.0.0", "epochUs": EPOCH_US, "deviceId": 6001},
        "modules": {
            "imu": {"id": IMU_ID, "size": IMU_SIZE, "rate": 100, "realtimeRate": 10},
            "gps": {"id": GPS_ID, "size": GPS_SIZE, "rate": 10, "realtimeRate": 10},
            "debug": {"id": 3, "size": 22, "rate": 1, "realtimeRate": 0},
        },
    }


def imu_frame(i: int) -> bytes:
    """Return the i-th IMU frame of a 10 Hz stream."""
    return struct.pack("<Bq7h", IMU_ID, EPOCH_US + i * 100_000, i, 2, 3, 4, 5, 6, 7)


def gps_frame(i: int) -> bytes:
    """Return the i-th GPS frame of a 10 Hz stream, offset by 5 ms from the IMU."""
    return struct.pack("<Bq6i", GPS_ID, EPOCH_US + i * 100_000 + 5_000, i, 0, 0, 0, 0, 0)


def payload(frames: list, run_id: int = 7) -> bytes:
    """Return a realtime payload: socket header followed by the frames."""
    return struct.pack("<BHH", 1, 6001, run_id) + b"".join(frames)


class TestPacking(unittest.TestCase):
    def test_pack_roundtrip_and_truncated_tail(self):
        items = [payload([imu_frame(i)]) for i in range(5)]
        blob = b"".join(pack_payload(p) for p in items)
        self.assertEqual(list(iter_payloads(blob)), items)
        self.assertEqual(list(iter_payloads(blob[:-3])), items[:-1])


class TestAssemble(unittest.TestCase):
    def setUp(self):
        self.header = make_header()
        self.expected = []
        for i in range(200):
            self.expected.extend([imu_frame(i), gps_frame(i)])

    def test_sorted_and_deduplicated(self):
        """Newest-first replay, two links chunking differently and duplicates give back the ordered stream."""
        frames = list(self.expected)
        chunks_a = [frames[i : i + 7] for i in range(0, len(frames), 7)]
        chunks_b = [frames[i : i + 11] for i in range(0, len(frames), 11)]
        random.Random(1).shuffle(chunks_a)
        chunks_b.reverse()
        blobs = [
            b"".join(pack_payload(payload(c)) for c in chunks_a),
            b"".join(pack_payload(payload(c)) for c in chunks_b[:15]),
        ]
        out, stats = assemble_frames(self.header, blobs)
        self.assertEqual(out, b"".join(self.expected))
        self.assertEqual(stats["frames"], 400)
        self.assertEqual(stats["duplicates"], sum(len(c) for c in chunks_b[:15]))
        self.assertEqual(stats["unsplit"], 0)

    def test_unsplit_remainder_and_time_frame_dropped(self):
        bad = payload([imu_frame(0), b"\x63" + b"\x00" * 10])
        time_frame = b"\x64" + b"\x00" * 12
        timed = struct.pack("<BHH", 1, 6001, 7) + time_frame + imu_frame(1)
        header = make_header()
        header["description"]["version"] = "v7.0.0"
        out, stats = assemble_frames(header, [pack_payload(bad) + pack_payload(timed)])
        self.assertEqual(out, imu_frame(0) + imu_frame(1))
        self.assertEqual(stats["unsplit"], 1)
        self.assertEqual(stats["time_frames"], 1)

    def test_empty(self):
        out, stats = assemble_frames(self.header, [b""])
        self.assertEqual(out, b"")
        self.assertEqual(stats["frames"], 0)


class TestDataTxt(unittest.TestCase):
    def test_description_uses_realtime_rate(self):
        header = make_header()
        description = build_description(header)
        self.assertEqual(description["modules"]["imu"]["rate"], 10)
        self.assertEqual(description["modules"]["debug"]["rate"], 1)
        self.assertEqual(header["modules"]["imu"]["rate"], 100)

    def test_loadfile_roundtrip(self):
        """The rebuilt file is read back by the SD loader: same header, calibration and frames."""
        header = make_header()
        calibration = {"imu": {"acc_offset": [1, 2, 3]}}
        frames = b"".join(imu_frame(i) + gps_frame(i) for i in range(300))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "wb") as f:
                f.write(build_data_txt(header, calibration, frames))
            loaded_header, loaded_calib, content = loadFile(path, use_s3=False)
        self.assertEqual(loaded_header, build_description(header))
        self.assertEqual(loaded_calib, calibration)
        self.assertEqual(content, frames)

    def test_decode_refuses_invalid_calibration(self):
        """A string coef in the header fails the decode with an explicit error, not float('') per frame."""
        calibration = {"imu": {"acc_x": {"coef": "", "offset": 0}}}
        frames = b"".join(imu_frame(i) for i in range(50))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "wb") as f:
                f.write(build_data_txt(make_header(), calibration, frames))
            with self.assertRaisesRegex(ValueError, r"^Invalid calibration imu\.acc_x\.coef: ''$"):
                decode(path, verbose=False, use_s3=False)


if __name__ == "__main__":
    unittest.main()
