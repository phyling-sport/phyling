import math
import os
import struct
import tempfile
import unittest

from phyling.decoder.decoder_utils import decode
from phyling.decoder.decoder_utils import loadAll
from phyling.decoder.decoder_utils import normalizeModuleFields
from phyling.decoder.realtime_record import build_data_txt

EPOCH_US = 1_790_000_000_000_000
POWER_ID = 1
POWER_SIZE = 14
SOCKET_HEADER = b"\x01\x3c\x00\x05\x00"


def make_header() -> dict:
    """Build a Maxi recDescription with one BLE powermeter module, as written by PowerMeterModule."""
    return {
        "description": {"version": "v7.0.7", "epochUs": EPOCH_US, "deviceId": 6001, "deviceType": "maxiphyling"},
        "modules": {
            "Assioma L": {
                "id": POWER_ID,
                "size": POWER_SIZE,
                "rate": 1.0,
                "realtimeRate": 1.0,
                "type": "powermeter",
                "description": [
                    ["id", "uint8", ""],
                    ["time", "uint64", ""],
                    ["power", "int16", "W"],
                    ["cadence", "uint16", "rpm"],
                    ["balance", "uint8", "%"],
                ],
            }
        },
    }


def power_frame(i: int, balance: int) -> bytes:
    """Return the i-th 1 Hz powermeter frame with the raw BLE balance byte."""
    return struct.pack("<BqhHB", POWER_ID, EPOCH_US + i * 1_000_000, 250, 90, balance)


class PowermeterBalanceTest(unittest.TestCase):
    """The raw BLE balance byte (left pedal % x 2, 0 = not available) is decoded to a percentage."""

    RAW_BALANCES = [93, 0, 100, 1]
    EXPECTED = [46.5, None, 50.0, 0.5]

    def assert_balances(self, values: list) -> None:
        """Compare decoded balances to EXPECTED, None standing for NaN."""
        self.assertEqual(len(values), len(self.EXPECTED))
        for value, expected in zip(values, self.EXPECTED):
            if expected is None:
                self.assertTrue(math.isnan(value), value)
            else:
                self.assertAlmostEqual(value, expected)

    def test_realtime_payload(self):
        """loadAll (realtime path) halves the balance and turns 0 into NaN, other fields untouched."""
        header = make_header()
        normalizeModuleFields(header)
        content = SOCKET_HEADER + b"".join(power_frame(i, b) for i, b in enumerate(self.RAW_BALANCES))
        frames = loadAll(header=header, content=content, check_higher_values=False, round_values=True)
        self.assert_balances([data["data"]["balance"] for data, _, _ in frames])
        self.assertEqual({data["data"]["power"] for data, _, _ in frames}, {250})
        self.assertEqual({data["data"]["cadence"] for data, _, _ in frames}, {90})

    def test_decode_file(self):
        """decode (SD / realtime record file) halves the balance and turns 0 into NaN."""
        frames = b"".join(power_frame(i, b) for i, b in enumerate(self.RAW_BALANCES))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "wb") as f:
                f.write(build_data_txt(make_header(), {}, frames))
            data = decode(path, verbose=False, use_s3=False)
        self.assert_balances(data["modules"]["Assioma L"]["data"]["balance"])


if __name__ == "__main__":
    unittest.main()
