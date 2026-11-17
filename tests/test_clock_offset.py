import logging
import unittest

import numpy as np

from phyling.decoder import calibration_use
from phyling.decoder.clock_offset import chunk_offset
from phyling.decoder.clock_offset import ClockOffsetTracker
from phyling.decoder.clock_offset import OFFSET_APPLIED
from phyling.decoder.clock_offset import OFFSET_OLD_MINI
from phyling.decoder.clock_offset import OFFSET_TOO_LARGE
from phyling.decoder.clock_offset import OFFSET_TOO_SHORT


def legacy_time_correction_v2(mod_data: dict, mod: str):
    """Verbatim copy of time_correction_v2 before the rule was shared, kept as the reference output."""
    MIN_TIME_DIFF_SEC = 15
    MAX_SHIFT_SEC = 3
    T = np.array(mod_data["data"]["T"])
    notifDiff = np.array(mod_data["data"]["notifDiff"])
    notifDiff[notifDiff < -MAX_SHIFT_SEC * 1e6] = max(notifDiff)
    ind = np.where(np.diff(T) > 3)[0] + 1
    ind = [0, *list(ind), len(T)]
    T_corr = []
    for i in range(len(ind) - 1):
        duration = T[ind[i + 1] - 1] - T[ind[i]]
        appliedDiff = 0
        if duration > MIN_TIME_DIFF_SEC:
            min_ = np.min(notifDiff[ind[i] : ind[i + 1]])
            max_ = np.max(notifDiff[ind[i] : ind[i + 1]])
            if not (min_ == 0 and max_ == 0):
                appliedDiff = min_ / 1e6 - 0.003
        if abs(appliedDiff) > MAX_SHIFT_SEC:
            appliedDiff = 0
        T_corr.extend(T[ind[i] : ind[i + 1]] + appliedDiff)
    return T_corr


def stream(t0: float, duration: float, rate: float, notif_us) -> tuple[np.ndarray, np.ndarray]:
    """Build a chunk of T (s) and notifDiff (us): notif_us is a constant or a callable of the sample index."""
    T = t0 + np.arange(int(duration * rate)) / rate
    nd = np.array([notif_us(i) if callable(notif_us) else notif_us for i in range(len(T))], dtype=np.int64)
    return T, nd


def fixture_chunks() -> tuple[np.ndarray, np.ndarray]:
    """Mini stream covering every rule: applied, too short, old mini, too large, outliers, reconnection gaps."""
    rng = np.random.default_rng(42)
    parts = [
        stream(0.0, 20, 50, lambda i: 25_000 + int(rng.integers(0, 8000))),  # applied
        stream(25.0, 10, 50, 40_000),  # too short
        stream(40.0, 20, 50, 0),  # old mini
        stream(65.0, 20, 50, 3_500_000),  # too large
        stream(90.0, 30, 50, lambda i: -5_000_000 if i % 97 == 0 else 12_000 + i),  # outliers
        stream(125.0, 16, 50, lambda i: -20_000 + i),  # negative offset
    ]
    T = np.concatenate([p[0] for p in parts])
    nd = np.concatenate([p[1] for p in parts])
    return T, nd


class TestChunkOffset(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(chunk_offset(20_000, 30_000, 15.0), (0.0, OFFSET_TOO_SHORT))
        self.assertEqual(chunk_offset(0, 0, 20.0), (0.0, OFFSET_OLD_MINI))
        offset, status = chunk_offset(20_000, 30_000, 15.01)
        self.assertEqual(status, OFFSET_APPLIED)
        self.assertAlmostEqual(offset, 0.017)
        offset, status = chunk_offset(3_100_000, 3_200_000, 20.0)
        self.assertEqual(status, OFFSET_TOO_LARGE)
        self.assertAlmostEqual(offset, 3.097)
        self.assertEqual(chunk_offset(0, 5, 20.0)[1], OFFSET_APPLIED)


class TestTimeCorrectionV2Unchanged(unittest.TestCase):
    def test_same_output_as_legacy(self):
        T, nd = fixture_chunks()
        mod_data = {"data": {"T": T.tolist(), "notifDiff": nd.tolist()}}
        logging.disable(logging.CRITICAL)
        try:
            expected = legacy_time_correction_v2(mod_data, "mini")
            got = calibration_use.time_correction_v2(mod_data, "mini")
        finally:
            logging.disable(logging.NOTSET)
        np.testing.assert_array_equal(np.array(got), np.array(expected))
        self.assertEqual(mod_data["data"]["notifDiff"], nd.tolist())  # input untouched

    def test_only_outliers_use_module_max(self):
        """An outlier-only chunk takes the module max: file behaviour kept as is."""
        T1, nd1 = stream(0.0, 20, 50, -5_000_000)
        T2, nd2 = stream(30.0, 20, 50, 30_000)
        mod_data = {"data": {"T": np.concatenate([T1, T2]), "notifDiff": np.concatenate([nd1, nd2])}}
        logging.disable(logging.CRITICAL)
        try:
            got = calibration_use.time_correction_v2(mod_data, "mini")
            expected = legacy_time_correction_v2(mod_data, "mini")
        finally:
            logging.disable(logging.NOTSET)
        np.testing.assert_array_equal(np.array(got), np.array(expected))


class TestClockOffsetTracker(unittest.TestCase):
    def run_tracker(self, T, nd, tracker=None):
        tracker = tracker or ClockOffsetTracker("mini")
        return tracker, np.array([tracker.update(float(t), float(n)) for t, n in zip(T, nd)])

    def test_raw_until_15s_then_running_min(self):
        T, nd = stream(0.0, 20, 50, lambda i: 30_000 - i)
        _, offsets = self.run_tracker(T, nd)
        self.assertTrue(np.all(offsets[T <= 15] == 0))
        after = T > 15
        np.testing.assert_allclose(offsets[after], (nd[after] / 1e6) - 0.003)  # running min = current value

    def test_matches_file_offset_at_chunk_end(self):
        T, nd = stream(0.0, 20, 50, lambda i: 25_000 + (i * 7919) % 8000)
        _, offsets = self.run_tracker(T, nd)
        logging.disable(logging.CRITICAL)
        try:
            file_T = calibration_use.time_correction_v2({"data": {"T": T, "notifDiff": nd}}, "mini")
        finally:
            logging.disable(logging.NOTSET)
        self.assertAlmostEqual(offsets[-1], file_T[-1] - T[-1])

    def test_gap_resets_chunk(self):
        T1, nd1 = stream(0.0, 20, 50, 10_000)
        T2, nd2 = stream(23.5, 20, 50, 50_000)
        tracker, offsets = self.run_tracker(np.concatenate([T1, T2]), np.concatenate([nd1, nd2]))
        n1 = len(T1)
        self.assertAlmostEqual(offsets[n1 - 1], 0.007)
        self.assertTrue(np.all(offsets[n1 : n1 + 15 * 50] == 0))  # new chunk raw again, no inherited min
        self.assertAlmostEqual(offsets[-1], 0.047)

    def test_small_gap_keeps_chunk(self):
        T1, nd1 = stream(0.0, 10, 50, 10_000)
        T2, nd2 = stream(12.0, 10, 50, 50_000)
        _, offsets = self.run_tracker(np.concatenate([T1, T2]), np.concatenate([nd1, nd2]))
        self.assertAlmostEqual(offsets[-1], 0.007)

    def test_too_large_ignored(self):
        T, nd = stream(0.0, 20, 50, 3_500_000)
        tracker, offsets = self.run_tracker(T, nd)
        self.assertTrue(np.all(offsets == 0))
        self.assertEqual(tracker.status, OFFSET_TOO_LARGE)

    def test_old_mini_zero_notif_diff(self):
        T, nd = stream(0.0, 20, 50, 0)
        tracker, offsets = self.run_tracker(T, nd)
        self.assertTrue(np.all(offsets == 0))
        self.assertEqual(tracker.status, OFFSET_OLD_MINI)

    def test_outliers_ignored(self):
        T, nd = stream(0.0, 20, 50, lambda i: -5_000_000 if i % 10 == 0 else 20_000)
        _, offsets = self.run_tracker(T, nd)
        self.assertAlmostEqual(offsets[-1], 0.017)

    def test_only_outliers_no_correction(self):
        T, nd = stream(0.0, 20, 50, -5_000_000)
        _, offsets = self.run_tracker(T, nd)
        self.assertTrue(np.all(offsets == 0))

    def test_late_sample_does_not_move_chunk_end(self):
        tracker = ClockOffsetTracker("mini")
        T, nd = stream(0.0, 20, 50, 10_000)
        self.run_tracker(T, nd, tracker)
        self.assertAlmostEqual(tracker.update(5.0, 10_000), 0.007)  # out-of-order sample: same chunk, same offset
        self.assertAlmostEqual(tracker.last_t, T[-1])


if __name__ == "__main__":
    unittest.main()
