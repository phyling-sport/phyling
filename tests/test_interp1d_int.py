import unittest

import numpy as np
import pandas as pd

from phyling.decoder.decoder import interp1d_


def _frame(n: int, **cols) -> pd.DataFrame:
    """Build a coarse DataFrame with a T column at 1 Hz."""
    return pd.DataFrame({"T": np.arange(n, dtype=np.float64), **cols})


class Interp1dIntegerColumnsTest(unittest.TestCase):
    """interp1d_ must round, not truncate, when casting interpolated columns back to integers."""

    def test_constant_label_stays_constant(self):
        """A constant int label resampled on a finer grid never drops to value - 1."""
        df = _frame(200, label=np.full(200, 3, dtype=np.int64), other=np.full(200, 7, dtype=np.int32))
        t_int = np.linspace(0, 199, 1990)
        out = interp1d_(df, ["label", "other"], t_int)
        self.assertEqual(set(out["label"]), {3})
        self.assertEqual(set(out["other"]), {7})

    def test_label_step_is_monotone(self):
        """A label going 3 -> 4 changes value once (no flicker)."""
        label = np.array([3] * 100 + [4] * 100, dtype=np.int64)
        out = interp1d_(_frame(200, label=label), ["label"], np.linspace(0, 199, 1990))
        changes = int((np.diff(out["label"].to_numpy()) != 0).sum())
        self.assertEqual(changes, 1)
        self.assertEqual(out["label"].iloc[0], 3)
        self.assertEqual(out["label"].iloc[-1], 4)

    def test_nan_outside_range_becomes_zero_for_int64(self):
        """NaN produced outside the source time range are still replaced by 0 for int columns."""
        df = _frame(5, label=np.full(5, 3, dtype=np.int64))
        out = interp1d_(df, ["label"], np.array([-1.0, 0.0, 2.5, 4.0, 5.0]))
        self.assertEqual(out["label"].tolist(), [0, 3, 3, 3, 0])
        self.assertEqual(out["label"].dtype, np.int64)

    def test_uint16_no_wrap(self):
        """uint16 values near the upper bound stay within range and are rounded."""
        values = np.array([65535, 65535, 65534, 0, 0], dtype=np.uint16)
        out = interp1d_(_frame(5, v=values), ["v"], np.linspace(0, 4, 17))
        self.assertEqual(out["v"].dtype, np.uint16)
        expected = np.rint(np.interp(np.linspace(0, 4, 17), np.arange(5), values.astype(float)))
        np.testing.assert_array_equal(out["v"].to_numpy(), expected.astype(np.uint16))

    def test_float_columns_untouched(self):
        """Float columns are not rounded."""
        df = _frame(3, f=np.array([0.0, 1.0, 2.0], dtype=np.float32))
        out = interp1d_(df, ["f"], np.array([0.25, 0.5]))
        np.testing.assert_allclose(out["f"].to_numpy(), [0.25, 0.5])


if __name__ == "__main__":
    unittest.main()
