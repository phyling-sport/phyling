import unittest

import numpy as np
import pandas as pd

from phyling.decoder.decoder import fuse


def _frames(x: str = "T") -> dict[str, pd.DataFrame]:
    """Two overlapping modules (100 Hz and 10 Hz), time in T (and x when x != T)."""
    t_imu = np.arange(0, 10, 0.01)
    t_gps = np.arange(0, 10, 0.1)
    res = {
        "imu": pd.DataFrame({"T": t_imu, "acc_x": np.sin(t_imu).astype("float32")}),
        "gps": pd.DataFrame({"T": t_gps, "speed": np.cos(t_gps), "fix": np.ones(len(t_gps), dtype="int64")}),
    }
    if x != "T":
        for frame in res.values():
            frame[x] = 1000.0 + frame["T"]
    return res


def _fuse_and_capture(res: dict, modules: list, **kwargs) -> tuple[pd.DataFrame, dict]:
    captured: dict = {}
    fused = fuse(res, modules, 100, native_frames=captured, **kwargs)
    return fused, captured


class CaptureNativeFramesTest(unittest.TestCase):
    """Test case for fuse(native_frames=...): captured names match the DataFrame fuse() returns."""

    def test_no_prefix(self):
        """Without prefix, columns keep their names."""
        fused, captured = _fuse_and_capture(_frames(), ["imu", "gps"])
        self.assertEqual(list(captured["imu"].columns), ["T", "acc_x"])
        self.assertEqual(list(captured["gps"].columns), ["T", "speed", "fix"])
        self.assertTrue({"T", "acc_x", "speed", "fix"} <= set(fused.columns))

    def test_prefix_multi_module(self):
        """With prefix and several modules, captured names are those of the fused df, T unprefixed."""
        fused, captured = _fuse_and_capture(_frames(), ["imu", "gps"], prefix=True)
        self.assertEqual(list(captured["imu"].columns), ["T", "imu.acc_x"])
        self.assertEqual(list(captured["gps"].columns), ["T", "gps.speed", "gps.fix"])
        captured_cols = set(captured["imu"].columns) | set(captured["gps"].columns)
        self.assertTrue(captured_cols <= set(fused.columns))

    def test_prefix_single_module(self):
        """fuse() returns a single module without prefix: the captured frame has no prefix, numeric columns only."""
        res = _frames()
        res["gps"]["label"] = "a"
        fused, captured = _fuse_and_capture(res, ["gps"], prefix=True)
        self.assertEqual(list(captured["gps"].columns), ["T", "speed", "fix"])
        self.assertTrue(set(captured["gps"].columns) <= set(fused.columns))

    def test_other_time_column(self):
        """With x != T, x and T keep their names (T is not renamed mod.T), the others match the fused df."""
        fused, captured = _fuse_and_capture(_frames("timestamp"), ["imu", "gps"], prefix=True, x="timestamp")
        self.assertEqual(list(captured["imu"].columns), ["T", "imu.acc_x", "timestamp"])
        self.assertEqual(list(captured["gps"].columns), ["T", "gps.speed", "gps.fix", "timestamp"])
        captured_cols = (set(captured["imu"].columns) | set(captured["gps"].columns)) - {"T"}
        self.assertTrue(captured_cols <= set(fused.columns))

    def test_other_time_column_single_module(self):
        """With x != T and a single module, the captured frame keeps T and the unprefixed names."""
        fused, captured = _fuse_and_capture(_frames("timestamp"), ["imu"], prefix=True, x="timestamp")
        self.assertEqual(list(captured["imu"].columns), ["T", "acc_x", "timestamp"])
        self.assertTrue(set(captured["imu"].columns) <= set(fused.columns))

    def test_t0_shift(self):
        """t0 shifts the time column of each captured frame like fuse() does on the fused df."""
        res = _frames()
        t_gps = res["gps"]["T"].to_numpy().copy()
        _, captured = _fuse_and_capture(res, ["imu", "gps"], t0={"imu": 1.0, "gps": 2.0})
        np.testing.assert_allclose(captured["gps"]["T"].to_numpy(), t_gps - 2.0)
        self.assertAlmostEqual(captured["imu"]["T"].iloc[0], -1.0)

    def test_dtypes_kept_and_no_copy(self):
        """Source dtypes are kept and the data is shared with the input (copy-on-write), until the input changes."""
        res = _frames()
        _, captured = _fuse_and_capture(res, ["imu", "gps"])
        frame = captured["gps"]
        self.assertEqual(frame["fix"].dtype, np.int64)
        self.assertEqual(captured["imu"]["acc_x"].dtype, np.float32)
        if int(pd.__version__.split(".")[0]) >= 3:  # copy-on-write is the default from pandas 3 only
            self.assertTrue(np.shares_memory(frame["speed"].to_numpy(), res["gps"]["speed"].to_numpy()))
        res["gps"].loc[0, "speed"] = 99.0
        self.assertNotEqual(frame["speed"].iloc[0], 99.0)

    def test_first_call_wins_and_default_none(self):
        """A second fuse() of the same key keeps the first capture; without native_frames nothing is captured."""
        res = _frames()
        captured: dict = {}
        fuse(res, ["imu", "gps"], 100, native_frames=captured)
        fuse({"gps": res["gps"].assign(speed=0.0), "imu": res["imu"]}, ["gps", "imu"], 100, native_frames=captured)
        self.assertNotEqual(float(captured["gps"]["speed"].abs().sum()), 0.0)
        fused = fuse(res, ["imu", "gps"], 100)
        self.assertIn("speed", fused)


if __name__ == "__main__":
    unittest.main()
