import unittest

import numpy as np

from phyling.decoder.calibration_use import calibration_1D
from phyling.decoder.calibration_use import calibration_1D_to_raw
from phyling.decoder.calibration_use import normalize_calibration


class Calibration1DToRawTest(unittest.TestCase):
    def test_round_trip(self):
        raw = np.array([-2.5, 0.0, 1.0, 1234.5])
        calibrated = calibration_1D(raw, coef=-3.2, offset=0.7)
        np.testing.assert_allclose(calibration_1D_to_raw(calibrated, coef=-3.2, offset=0.7), raw, atol=1e-12)

    def test_identity_when_absent(self):
        self.assertEqual(calibration_1D_to_raw(4.2), 4.2)

    def test_null_coef(self):
        with self.assertRaises(ValueError):
            calibration_1D_to_raw(1.0, coef=0)


class NormalizeCalibrationTest(unittest.TestCase):
    """normalize_calibration validates coef/offset once, before the frames are decoded."""

    VALID = {
        "Press-Sure": {"data": {"texisense_calib_base64": "QUJD" * 100, "texisense_metadata": {"rows": 16}}},
        "rene-d": {"adc_0": {"coef": 160.78, "offset": -0.215}},
        "pedalier": {"algo": {"theta0": 0, "magHysteresis": 0.2}, "gyro_z": {"coef": -1, "offset": 0}},
        "imu": {"acc": {"coef": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "offset": [0.1, 0.2, 0.3]}, "mode": "x"},
    }

    def test_valid_calibration_unchanged(self):
        """A valid calibration comes back equal, texisense base64 and algo params untouched."""
        out = normalize_calibration(self.VALID)
        self.assertEqual(out, self.VALID)
        texisense = self.VALID["Press-Sure"]["data"]
        self.assertIs(out["Press-Sure"]["data"]["texisense_calib_base64"], texisense["texisense_calib_base64"])

    def test_none_and_empty(self):
        """No calibration stays as is."""
        self.assertIsNone(normalize_calibration(None))
        self.assertEqual(normalize_calibration({}), {})

    def test_numeric_string_converted(self):
        """A quoted number is converted to float, the input is not modified."""
        calib = {"rene-d": {"adc_0": {"coef": "0.5", "offset": " -2 "}}}
        out = normalize_calibration(calib)
        self.assertEqual(out["rene-d"]["adc_0"], {"coef": 0.5, "offset": -2.0})
        self.assertEqual(calib["rene-d"]["adc_0"]["coef"], "0.5")

    def test_empty_string_refused(self):
        """An empty coef raises with the module, field and key named."""
        with self.assertRaisesRegex(ValueError, r"^Invalid calibration rene-d\.adc_0\.coef: ''$"):
            normalize_calibration({"rene-d": {"adc_0": {"coef": "", "offset": 1}}})

    def test_text_refused(self):
        """A non-numeric offset raises."""
        with self.assertRaisesRegex(ValueError, r"Invalid calibration rene-g\.adc_0\.offset: 'abc'"):
            normalize_calibration({"rene-g": {"adc_0": {"coef": 1, "offset": "abc"}}})

    def test_nested_3d_lists(self):
        """3D matrices are walked: numeric strings converted, a bad string refused."""
        calib = {"imu": {"acc": {"coef": [[1, "0"], ["2.5", 3]], "offset": ["1", 2, 3]}}}
        out = normalize_calibration(calib)
        self.assertEqual(out["imu"]["acc"], {"coef": [[1, 0.0], [2.5, 3]], "offset": [1.0, 2, 3]})
        with self.assertRaisesRegex(ValueError, r"Invalid calibration imu\.gyro\.coef: ''"):
            normalize_calibration({"imu": {"gyro": {"coef": [[1, 0, 0], [0, "", 0], [0, 0, 1]]}}})

    def test_not_an_object_refused(self):
        """A calibration that is not a JSON object raises."""
        with self.assertRaisesRegex(ValueError, "Invalid calibration"):
            normalize_calibration([1, 2])
