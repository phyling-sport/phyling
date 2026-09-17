import base64
import struct
import unittest

import numpy as np

from phyling.decoder.calib_texisense import _phase_lut
from phyling.decoder.calib_texisense import apply_texisense_calibration
from phyling.decoder.calib_texisense import CHUNK_SIZE
from phyling.decoder.calib_texisense import generate_lut
from phyling.decoder.calib_texisense import generate_texisense_calibration_data
from phyling.decoder.calib_texisense import get_calibration_fingerprint
from phyling.decoder.calib_texisense import NUM_SENSORS
from phyling.decoder.calib_texisense import RAW_LEVELS
from phyling.decoder.calib_texisense import read_lut_metadata

MAX_PRESSURE = 64000
NUM_STEPS = 12  # deliberately not the 45 steps of the mat we calibrated against


def build_buffer(num_steps=NUM_STEPS, max_pressure=MAX_PRESSURE, saturate=False):
    """Build a synthetic calibration blob: a pressure sweep up then back down, saturating sensors.

    Each sensor gets its own gain so a lookup table mixed up between sensors shows up in the tests.
    The descending phase reads higher than the ascending one at the same pressure, as a real mat
    does, and its steps fall between the ascending ones so the two phases cannot be merged naively.

    Parameters:
        num_steps (int): number of steps of the ascending phase
        max_pressure (int): pressure of the highest step, in Pascal
        saturate (bool): if True the top steps clip at 255, as a real mat does under heavy load

    Returns:
        tuple: (buffer, pressures) with buffer the encoded blob and pressures the steps it holds
    """
    up = [round(max_pressure * i / (num_steps - 1)) for i in range(num_steps)]
    down = [round(p * 0.96) for p in reversed(up[:-1])]
    pressures = up + down

    gains = 1.0 + 0.5 * np.arange(NUM_SENSORS) / NUM_SENSORS
    buffer = b""
    for step, pressure in enumerate(pressures):
        ratio = pressure / max_pressure
        hysteresis = 1.0 if step < num_steps else 1.08
        full_scale = 340 if saturate else 170
        responses = (
            np.round(full_scale * gains * hysteresis * np.sqrt(ratio))
            .clip(0, 255)
            .astype(np.uint8)
        )
        buffer += struct.pack(">H", pressure) + responses.tobytes()
    return buffer, pressures


class CalibTexisenseTest(unittest.TestCase):
    def setUp(self):
        self.buffer, self.pressures = build_buffer()

    def test_parses_any_step_count(self):
        """The step count comes from the buffer length, it is not assumed to be 45."""
        pressures, raws = generate_texisense_calibration_data(self.buffer)
        self.assertEqual(len(pressures), len(self.pressures))
        self.assertEqual(raws.shape, (len(self.pressures), NUM_SENSORS))

    def test_pressure_is_big_endian(self):
        """A little-endian read would divide every step by 256 and cap the mat at 250 Pa."""
        pressures, _ = generate_texisense_calibration_data(self.buffer)
        self.assertEqual(pressures.max(), MAX_PRESSURE)
        self.assertListEqual(list(pressures), self.pressures)

    def test_rejects_truncated_buffer(self):
        with self.assertRaises(ValueError):
            generate_texisense_calibration_data(self.buffer[: CHUNK_SIZE + 17])

    def test_lut_is_monotonic(self):
        """Regression: merging both phases by pressure fed np.interp a non-monotonic axis."""
        lut = generate_lut(*generate_texisense_calibration_data(self.buffer))
        self.assertTrue(np.all(np.diff(lut, axis=1) >= -1e-3))

    def test_zero_raw_gives_zero_pressure(self):
        """An unloaded cell must read exactly 0 Pa, not a floor left by the interpolation."""
        lut = generate_lut(*generate_texisense_calibration_data(self.buffer))
        self.assertTrue(np.all(lut[:, 0] == 0))

    def test_sensor_mute_over_the_first_steps_still_reads_zero(self):
        """Regression: a cell answering 0 up to 1500 Pa inherited the mean of those steps as its floor.

        The synthetic sweep never exercises this - its sqrt response leaves 0 only at the 0 Pa step -
        yet a lightly loaded cell of a real mat stays mute over the first steps of the sweep.
        """
        pressures = np.array([0, 100, 300, 700, 1500, 3000, 6000])
        raws = np.array([0, 0, 0, 0, 12, 40, 90])
        lut = _phase_lut(pressures, raws)
        self.assertEqual(lut[0], 0)
        self.assertTrue(np.all(np.diff(lut) >= 0))

    def test_calibration_steps_are_recovered(self):
        """Feeding back the response recorded at a step must return that step's pressure."""
        pressures, raws = generate_texisense_calibration_data(self.buffer)
        lut = generate_lut(pressures, raws)
        step = len(self.pressures) // 3
        recovered = lut[np.arange(NUM_SENSORS), raws[step]]
        self.assertLess(
            np.median(np.abs(recovered - pressures[step])), 0.1 * MAX_PRESSURE
        )

    def test_apply_transposes_the_matrix(self):
        """The mat was already laid out the right way up before the calibration was fixed."""
        pressures, raws = generate_texisense_calibration_data(self.buffer)
        step = len(self.pressures) // 2
        raw_matrix = raws[step].reshape(32, 32)
        out = apply_texisense_calibration(raw_matrix, base64.b64encode(self.buffer))
        lut = generate_lut(pressures, raws)
        expected = lut[np.arange(NUM_SENSORS), raws[step]].reshape(32, 32).transpose()
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_extrapolates_past_the_last_step(self):
        """Above the highest calibrated response the curve must keep rising, not plateau."""
        lut = generate_lut(*generate_texisense_calibration_data(self.buffer))
        self.assertGreater(lut[:, 255].min(), lut[:, 200].min())

    def test_saturated_sensors_stay_monotonic(self):
        """Regression: averaging the steps a saturated sensor shares made the curve dip, then
        extrapolate to large negative pressures."""
        buffer, _ = build_buffer(saturate=True)
        lut = generate_lut(*generate_texisense_calibration_data(buffer))
        self.assertTrue(np.all(np.diff(lut, axis=1) >= -1e-3))
        self.assertGreaterEqual(lut.min(), 0)


FULL_METADATA = {
    "metadata_version": 1,
    "serial": "N0000HG",
    "calib_date": "2026-03-14T09:21:05",
    "threshold_pa": 1500,
    "sensor_width": 470,
    "sensor_length": 470,
    "step_width_01mm": 125,
    "step_length_01mm": 125,
    "weight_asc": 1,
    "weight_desc": 1,
    "calib_method": 2,
    "mirror_cols": False,
    "mirror_rows": True,
    "swap_rows_cols": False,
}


class CalibTexisenseMetadataTest(unittest.TestCase):
    def setUp(self):
        self.buffer, self.pressures = build_buffer()
        self.b64 = base64.b64encode(self.buffer)
        self.pressures, self.raws = generate_texisense_calibration_data(self.buffer)
        self.step = len(self.pressures) // 2
        self.raw_matrix = self.raws[self.step].reshape(32, 32)

    def _legacy_output(self):
        """The output of the calibration as it behaved before metadata existed."""
        lut = generate_lut(self.pressures, self.raws)
        return (
            lut[np.arange(NUM_SENSORS), self.raws[self.step]]
            .reshape(32, 32)
            .transpose()
        )

    def test_missing_metadata_keeps_legacy_output(self):
        """Every stored record and every mat not read since the firmware reports metadata has none."""
        expected = self._legacy_output()
        for metadata in (
            None,
            {},
            {"serial": "N0000HG", "calib_date": "2026-03-14T09:21:05"},
        ):
            out = apply_texisense_calibration(
                self.raw_matrix, self.b64, metadata=metadata
            )
            np.testing.assert_array_equal(out, expected)

    def test_any_single_key_may_be_missing(self):
        """No key is mandatory: dropping one must never raise nor produce a broken matrix."""
        for dropped in FULL_METADATA:
            metadata = {k: v for k, v in FULL_METADATA.items() if k != dropped}
            out = apply_texisense_calibration(
                self.raw_matrix, self.b64, metadata=metadata
            )
            self.assertEqual(out.shape, (32, 32), msg=f"dropping {dropped}")
            self.assertTrue(np.all(np.isfinite(out)), msg=f"dropping {dropped}")

    def test_missing_weight_falls_back_to_one(self):
        metadata = {k: v for k, v in FULL_METADATA.items() if k != "weight_asc"}
        out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
        expected = apply_texisense_calibration(
            self.raw_matrix, self.b64, metadata=FULL_METADATA
        )
        np.testing.assert_array_equal(out, expected)

    def test_weights_move_the_curve_between_the_two_phases(self):
        asc_only = generate_lut(self.pressures, self.raws, weight_asc=1, weight_desc=0)
        desc_only = generate_lut(self.pressures, self.raws, weight_asc=0, weight_desc=1)
        even = generate_lut(self.pressures, self.raws)
        self.assertGreater(np.abs(asc_only - desc_only).max(), 0.01 * MAX_PRESSURE)
        np.testing.assert_allclose(even, (asc_only + desc_only) / 2, rtol=1e-5)
        np.testing.assert_allclose(
            generate_lut(self.pressures, self.raws, weight_asc=3, weight_desc=1),
            (3 * asc_only + desc_only) / 4,
            rtol=1e-5,
        )

    def test_both_weights_zero_falls_back_to_an_even_average(self):
        """Honouring a 0/0 weighting would divide by zero; the even average is the sane fallback."""
        self.assertEqual(
            read_lut_metadata({"weight_asc": 0, "weight_desc": 0}), (1.0, 1.0, 0.0)
        )

    def test_threshold_zeroes_below_and_keeps_above(self):
        threshold = 5000
        # a sweep over the whole raw range, so cells land on both sides of the threshold
        raw_matrix = (np.arange(NUM_SENSORS) % RAW_LEVELS).reshape(32, 32)
        out = apply_texisense_calibration(
            raw_matrix, self.b64, metadata={"threshold_pa": threshold}
        )
        reference = apply_texisense_calibration(raw_matrix, self.b64)
        self.assertTrue(np.all((out == 0) | (out >= threshold)))
        np.testing.assert_array_equal(
            out[reference >= threshold], reference[reference >= threshold]
        )
        self.assertTrue(np.all(out[reference < threshold] == 0))
        self.assertGreater((reference < threshold).sum(), 0)

    def test_missing_threshold_applies_no_floor(self):
        metadata = {k: v for k, v in FULL_METADATA.items() if k != "threshold_pa"}
        out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
        np.testing.assert_array_equal(out, self._legacy_output())

    def test_orientation_ignores_the_metadata_flags(self):
        """The mat was already displayed the right way up: the flags are stored, never applied."""
        expected = self._legacy_output()
        for flags in (
            {},
            {"swap_rows_cols": True},
            {"mirror_rows": True},
            {"mirror_cols": True},
        ):
            out = apply_texisense_calibration(
                self.raw_matrix, self.b64, metadata=dict(FULL_METADATA, **flags)
            )
            np.testing.assert_array_equal(
                out,
                apply_texisense_calibration(
                    self.raw_matrix, self.b64, metadata=FULL_METADATA
                ),
            )
        np.testing.assert_array_equal(
            apply_texisense_calibration(self.raw_matrix, self.b64), expected
        )

    def test_cache_key_separates_different_metadata(self):
        """Two mats sharing a buffer but not their metadata must not share a lookup table."""
        keys = {
            get_calibration_fingerprint(self.buffer),
            get_calibration_fingerprint(self.buffer, weight_asc=3),
            get_calibration_fingerprint(self.buffer, weight_desc=3),
            get_calibration_fingerprint(self.buffer, threshold_pa=1500),
        }
        self.assertEqual(len(keys), 4)

    def test_orientation_stays_out_of_the_cache_key(self):
        """Orientation is applied after the lookup, it does not shape the table."""
        self.assertEqual(
            get_calibration_fingerprint(self.buffer),
            get_calibration_fingerprint(self.buffer, 1.0, 1.0, 0.0),
        )

    def test_mat_dimensions_are_ignored(self):
        """sensor_width/length are millimetres, not cell counts: a real mat reports 470 for a 32x32."""
        metadata = dict(FULL_METADATA, sensor_width=470, sensor_length=470)
        out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
        expected = apply_texisense_calibration(
            self.raw_matrix, self.b64, metadata=FULL_METADATA
        )
        self.assertEqual(out.shape, (32, 32))
        np.testing.assert_array_equal(out, expected)
