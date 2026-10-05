import unittest

import numpy as np

from phyling.decoder.calibration_use import calibration_1D
from phyling.decoder.calibration_use import calibration_1D_to_raw


class Calibration1DToRawTest(unittest.TestCase):
    def test_round_trip(self):
        raw = np.array([-2.5, 0.0, 1.0, 1234.5])
        calibrated = calibration_1D(raw, coef=-3.2, offset=0.7)
        np.testing.assert_allclose(
            calibration_1D_to_raw(calibrated, coef=-3.2, offset=0.7), raw, atol=1e-12
        )

    def test_identity_when_absent(self):
        self.assertEqual(calibration_1D_to_raw(4.2), 4.2)

    def test_null_coef(self):
        with self.assertRaises(ValueError):
            calibration_1D_to_raw(1.0, coef=0)
