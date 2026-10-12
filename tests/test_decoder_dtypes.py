import unittest

import numpy as np

from phyling.decoder.decoder import datamodule2df

GPSTIME = [1712345678, 1712345679, 1712345680]


class DatamoduleDtypesTest(unittest.TestCase):
    """datamodule2df column dtypes: float32 by default, wider where float32 cannot hold the value."""

    def test_gps_dtypes(self):
        """gpstime keeps its one-second steps (float32 resolves 128 s at 1.7e9), other fields stay float32."""
        data = {
            "modules": {
                "gps": {
                    "T": [0.0, 1.0, 2.0],
                    "data": {
                        "gpstime": GPSTIME,
                        "gpstimeUs": [t * 1_000_000 for t in GPSTIME],
                        "latitude": [45.1, 45.2, 45.3],
                        "speed": [1.0, 2.0, 3.0],
                    },
                }
            }
        }
        df = datamodule2df(data, "gps")
        self.assertEqual(df["gpstime"].dtype, np.float64)
        self.assertEqual(df["gpstime"].tolist(), GPSTIME)
        self.assertEqual(df["gpstimeUs"].dtype, np.int64)
        self.assertEqual(df["latitude"].dtype, np.float64)
        self.assertEqual(df["speed"].dtype, np.float32)
        self.assertEqual(df["T"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
