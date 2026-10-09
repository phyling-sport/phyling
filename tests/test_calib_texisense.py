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
from phyling.decoder.calib_texisense import smooth_pressure_map
from phyling.decoder.calib_texisense import TEXISENSE_PRESSURE_UNIT_PA

MAX_UNITS = 650  # highest step, in gf/cm2: its word has a non-zero high byte
MAX_PRESSURE = MAX_UNITS * TEXISENSE_PRESSURE_UNIT_PA
NUM_STEPS = 12  # deliberately not the 45 steps of the mat we calibrated against


def build_buffer(num_steps=NUM_STEPS, max_units=MAX_UNITS, saturate=False):
    """Build a synthetic calibration blob: a pressure sweep up then back down, saturating sensors.

    Each sensor gets its own gain so a lookup table mixed up between sensors shows up in the tests.
    The descending phase reads higher than the ascending one at the same pressure, as a real mat
    does, and its steps fall between the ascending ones so the two phases cannot be merged naively.

    Parameters:
        num_steps (int): number of steps of the ascending phase
        max_units (int): pressure of the highest step, in gf/cm2 as the mat stores it
        saturate (bool): if True the top steps clip at 255, as a real mat does under heavy load

    Returns:
        tuple: (buffer, pressures) with buffer the encoded blob and pressures its steps in Pascal
    """
    up = [round(max_units * i / (num_steps - 1)) for i in range(num_steps)]
    down = [round(p * 0.96) for p in reversed(up[:-1])]
    units = up + down

    gains = 1.0 + 0.5 * np.arange(NUM_SENSORS) / NUM_SENSORS
    buffer = b""
    for step, unit in enumerate(units):
        ratio = unit / max_units
        hysteresis = 1.0 if step < num_steps else 1.08
        full_scale = 340 if saturate else 170
        responses = np.round(full_scale * gains * hysteresis * np.sqrt(ratio)).clip(0, 255).astype(np.uint8)
        buffer += struct.pack("<H", unit) + responses.tobytes()
    return buffer, [u * TEXISENSE_PRESSURE_UNIT_PA for u in units]


class CalibTexisenseTest(unittest.TestCase):
    def setUp(self):
        self.buffer, self.pressures = build_buffer()

    def test_parses_any_step_count(self):
        """The step count comes from the buffer length, it is not assumed to be 45."""
        pressures, raws = generate_texisense_calibration_data(self.buffer)
        self.assertEqual(len(pressures), len(self.pressures))
        self.assertEqual(raws.shape, (len(self.pressures), NUM_SENSORS))

    def test_pressure_is_little_endian_gf_per_cm2(self):
        """Each step is a little-endian word in gf/cm2, as the vendor per-cell export shows.

        The former big-endian read in Pascal made the mat 256 / 98.0665 = 2.61 times too heavy.
        """
        pressures, _ = generate_texisense_calibration_data(self.buffer)
        self.assertAlmostEqual(pressures.max(), 650 * 98.0665)
        np.testing.assert_allclose(pressures, self.pressures)
        big_endian = struct.unpack(">H", struct.pack("<H", MAX_UNITS))[0]
        self.assertNotAlmostEqual(pressures.max(), big_endian)

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
        self.assertLess(np.median(np.abs(recovered - pressures[step])), 0.1 * MAX_PRESSURE)

    def test_apply_reorders_then_smooths_the_matrix(self):
        """The lookup is per sensor; the map is then mirrored into Texisense order and smoothed."""
        pressures, raws = generate_texisense_calibration_data(self.buffer)
        step = len(self.pressures) // 2
        raw_matrix = raws[step].reshape(32, 32)
        out = apply_texisense_calibration(raw_matrix, base64.b64encode(self.buffer))
        lut = generate_lut(pressures, raws)
        mapped = lut[np.arange(NUM_SENSORS), raws[step]].reshape(32, 32)[::-1]
        np.testing.assert_allclose(out, smooth_pressure_map(mapped), rtol=1e-5)

    def test_output_is_in_texisense_order(self):
        """Sensor s lands on Texisense cell (r, c) = (31 - s // 32, s % 32), front-left first."""
        pressures, raws = generate_texisense_calibration_data(self.buffer)
        lut = generate_lut(pressures, raws)
        for sensor in (0, 31, 32 * 31, 10 * 32 + 12):
            raw_matrix = np.zeros((32, 32), dtype=int)
            raw_matrix.flat[sensor] = 200
            out = apply_texisense_calibration(raw_matrix, base64.b64encode(self.buffer))
            r, c = 31 - sensor // 32, sensor % 32
            self.assertEqual(np.unravel_index(np.argmax(out), out.shape), (r, c))
            weight = 6 if r in (0, 31) and c in (0, 31) else 11
            weight = 8 if weight == 11 and (r in (0, 31) or c in (0, 31)) else weight
            self.assertAlmostEqual(out[r, c], 3 * lut[sensor, 200] / weight, delta=0.1)

    def test_extrapolates_in_proportion_past_the_bench_ceiling(self):
        """Above its highest bench response a sensor reads P_top x raw / raw_top.

        Prolonging the last segment instead gave 97 398 Pa on a cell the vendor reads 33 346 Pa.
        Here the last segments would give 7000 Pa (ascending) and 4500 Pa (descending) at raw 200.
        """
        pressures = np.array([0.0, 1000.0, 2000.0, 1000.0, 0.0])
        raws = np.tile(np.array([0, 80, 100, 60, 0])[:, None], (1, NUM_SENSORS))
        lut = generate_lut(pressures, raws)
        self.assertAlmostEqual(lut[0, 100], 2000.0, places=2)
        self.assertAlmostEqual(lut[0, 150], 3000.0, places=2)
        self.assertAlmostEqual(lut[0, 200], 4000.0, places=2)
        self.assertTrue(np.all(np.diff(lut, axis=1) >= 0))

    def test_extrapolation_is_anchored_on_the_top_step(self):
        """Review probe: raw 100 at the 3000 Pa peak, 110 at 2000 Pa on the way down.

        The line P_top x raw / raw_top starts at raw_top = 100, the response at the peak, not at
        the highest response of the sweep: raw 110 reads 3300 Pa, raw 111 3330 Pa, no plateau.
        """
        pressures = np.array([0.0, 3000.0, 2000.0, 0.0])
        raws = np.tile(np.array([0, 100, 110, 0])[:, None], (1, NUM_SENSORS))
        lut = generate_lut(pressures, raws)
        self.assertAlmostEqual(lut[0, 100], 3000.0, places=2)
        self.assertAlmostEqual(lut[0, 110], 3300.0, places=2)
        self.assertAlmostEqual(lut[0, 111], 3330.0, places=2)
        self.assertTrue(np.all(np.diff(lut[0, 1:]) > 0))

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
        """The output of the calibration as it behaves without metadata."""
        lut = generate_lut(self.pressures, self.raws)
        return smooth_pressure_map(lut[np.arange(NUM_SENSORS), self.raws[self.step]].reshape(32, 32)[::-1])

    def test_missing_metadata_keeps_legacy_output(self):
        """Every stored record and every mat not read since the firmware reports metadata has none."""
        expected = self._legacy_output()
        for metadata in (
            None,
            {},
            {"serial": "N0000HG", "calib_date": "2026-03-14T09:21:05"},
        ):
            out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
            np.testing.assert_array_equal(out, expected)

    def test_any_single_key_may_be_missing(self):
        """No key is mandatory: dropping one must never raise nor produce a broken matrix."""
        for dropped in FULL_METADATA:
            metadata = {k: v for k, v in FULL_METADATA.items() if k != dropped}
            out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
            self.assertEqual(out.shape, (32, 32), msg=f"dropping {dropped}")
            self.assertTrue(np.all(np.isfinite(out)), msg=f"dropping {dropped}")

    def test_missing_weight_falls_back_to_one(self):
        metadata = {k: v for k, v in FULL_METADATA.items() if k != "weight_asc"}
        out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
        expected = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=FULL_METADATA)
        np.testing.assert_array_equal(out, expected)

    def test_weights_move_the_curve_between_the_two_phases(self):
        asc_only = generate_lut(self.pressures, self.raws, weight_asc=1, weight_desc=0)
        desc_only = generate_lut(self.pressures, self.raws, weight_asc=0, weight_desc=1)
        even = generate_lut(self.pressures, self.raws)
        weighted = generate_lut(self.pressures, self.raws, weight_asc=3, weight_desc=1)
        # compare within the range both phases calibrate, before any extrapolation
        top = int(np.argmax(self.pressures))
        in_range = (
            np.arange(RAW_LEVELS)[None, :]
            <= np.minimum(self.raws[: top + 1].max(axis=0), self.raws[top:].max(axis=0))[:, None]
        )
        self.assertGreater(np.abs(asc_only - desc_only).max(), 0.01 * MAX_PRESSURE)
        np.testing.assert_allclose(even[in_range], ((asc_only + desc_only) / 2)[in_range], rtol=1e-5)
        np.testing.assert_allclose(weighted[in_range], ((3 * asc_only + desc_only) / 4)[in_range], rtol=1e-5)

    def test_both_weights_zero_falls_back_to_an_even_average(self):
        """Honouring a 0/0 weighting would divide by zero; the even average is the sane fallback."""
        self.assertEqual(read_lut_metadata({"weight_asc": 0, "weight_desc": 0}), (1.0, 1.0, 0.0))

    def test_threshold_zeroes_below_and_keeps_above(self):
        threshold = 5000
        floored = generate_lut(self.pressures, self.raws, threshold_pa=threshold)
        reference = generate_lut(self.pressures, self.raws)
        self.assertTrue(np.all((floored == 0) | (floored >= threshold)))
        above = reference >= threshold
        np.testing.assert_array_equal(floored[above], reference[above])
        self.assertTrue(np.all(floored[~above] == 0))
        self.assertGreater((~above).sum(), 0)

    def test_threshold_applies_before_smoothing(self):
        """The floor acts on each cell in true Pascal, then the smoothing spreads what is left.

        A loaded cell surrounded by cells under the floor leaves a halo of 1/11 of its pressure,
        below the floor: flooring after the smoothing would erase it, as the vendor does not.
        """
        threshold = 5000
        lut = generate_lut(self.pressures, self.raws)
        sensor = 10 * 32 + 12
        high = int(np.argmax(lut[sensor] >= 4 * threshold))
        low = int(np.argmax(lut[sensor] >= threshold / 2))
        self.assertLess(lut[sensor, low], threshold)
        raw_matrix = np.full((32, 32), low)
        raw_matrix[10, 12] = high
        out = apply_texisense_calibration(raw_matrix, self.b64, metadata={"threshold_pa": threshold})
        floored = generate_lut(self.pressures, self.raws, threshold_pa=threshold)
        mapped = floored[np.arange(NUM_SENSORS), raw_matrix.flatten()]
        np.testing.assert_allclose(out, smooth_pressure_map(mapped.reshape(32, 32)[::-1]), rtol=1e-5)
        # the loaded sensor 10 * 32 + 12 lands on Texisense cell (21, 12)
        self.assertAlmostEqual(out[21, 12], 3 * floored[sensor, high] / 11, delta=0.1)
        self.assertAlmostEqual(out[20, 12], floored[sensor, high] / 11, delta=0.1)
        self.assertGreater(out[20, 12], 0)
        self.assertLess(out[20, 12], threshold)
        self.assertEqual(out[0, 0], 0)

    def test_missing_threshold_applies_no_floor(self):
        metadata = {k: v for k, v in FULL_METADATA.items() if k != "threshold_pa"}
        out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=metadata)
        np.testing.assert_array_equal(out, self._legacy_output())

    def test_orientation_ignores_the_metadata_flags(self):
        """The Texisense order comes from the vendor export: the flags are stored, never applied."""
        expected = self._legacy_output()
        for flags in (
            {},
            {"swap_rows_cols": True},
            {"mirror_rows": True},
            {"mirror_cols": True},
        ):
            out = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=dict(FULL_METADATA, **flags))
            np.testing.assert_array_equal(
                out,
                apply_texisense_calibration(self.raw_matrix, self.b64, metadata=FULL_METADATA),
            )
        np.testing.assert_array_equal(apply_texisense_calibration(self.raw_matrix, self.b64), expected)

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
        expected = apply_texisense_calibration(self.raw_matrix, self.b64, metadata=FULL_METADATA)
        self.assertEqual(out.shape, (32, 32))
        np.testing.assert_array_equal(out, expected)


class CentreColumnTest(unittest.TestCase):
    """Column 15, zeroed by the vendor software, is kept and smoothed like any other."""

    def test_load_on_the_centre_column_is_kept_and_spreads(self):
        buffer, _ = build_buffer()
        pressures, raws = generate_texisense_calibration_data(buffer)
        lut = generate_lut(pressures, raws)
        raw_matrix = np.zeros((32, 32), dtype=int)
        sensor = (31 - 20) * 32 + 15
        raw_matrix.flat[sensor] = 200
        out = apply_texisense_calibration(raw_matrix, base64.b64encode(buffer))
        self.assertAlmostEqual(out[20, 15], 3 * lut[sensor, 200] / 11, delta=0.1)
        for r in (19, 20, 21):
            for c in (14, 16):
                self.assertAlmostEqual(out[r, c], lut[sensor, 200] / 11, delta=0.1)
        self.assertEqual(np.count_nonzero(out), 9)


class SmoothingTest(unittest.TestCase):
    """The vendor 3x3 smoothing: kernel [[1, 1, 1], [1, 3, 1], [1, 1, 1]], edge-renormalised."""

    def test_centre_cell_spreads_over_its_neighbours(self):
        mat = np.zeros((32, 32))
        mat[10, 20] = 1100.0
        out = smooth_pressure_map(mat)
        self.assertAlmostEqual(out[10, 20], 300.0, places=3)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    self.assertAlmostEqual(out[10 + dr, 20 + dc], 100.0, places=3)
        self.assertAlmostEqual(out.sum(), 1100.0, places=2)
        self.assertEqual(np.count_nonzero(out), 9)

    def test_edges_divide_by_the_weights_inside_the_mat(self):
        mat = np.zeros((32, 32))
        mat[0, 0] = 600.0
        out = smooth_pressure_map(mat)
        # corner: 3 + 3 neighbours inside the mat = 6
        self.assertAlmostEqual(out[0, 0], 300.0, places=3)
        # edge cell next to the corner: 3 + 5 neighbours inside the mat = 8
        self.assertAlmostEqual(out[0, 1], 75.0, places=3)
        self.assertAlmostEqual(out[1, 0], 75.0, places=3)
        # inner cell next to the corner: full kernel, 11
        self.assertAlmostEqual(out[1, 1], 600.0 / 11, places=3)

    def test_uniform_map_stays_uniform(self):
        """Renormalising on the border keeps a uniform load uniform, corners included."""
        out = smooth_pressure_map(np.full((32, 32), 2500.0))
        np.testing.assert_allclose(out, 2500.0, rtol=1e-6)

    def test_output_is_float32(self):
        self.assertEqual(smooth_pressure_map(np.ones((32, 32))).dtype, np.float32)
