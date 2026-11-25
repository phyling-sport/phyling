import unittest

import numpy as np
import pandas as pd

from phyling.decoder.decoder import fuse
from phyling.decoder.decoder import fuse_data


def _frame(fs: float, label: str) -> pd.DataFrame:
    """One module at fs Hz over 1 s, with an ISO "time" text column like a re-uploaded Phyling CSV export."""
    t = np.arange(0, 1, 1 / fs)
    return pd.DataFrame({"T": t, "time": [f"2026-11-20T13:05:46.{i:03d}Z" for i in range(len(t))], label: t * 2})


class FuseTextColumnsTest(unittest.TestCase):
    """A text column cannot be interpolated: fuse_data leaves it out of the interpolated frames only."""

    def test_text_column_of_df2_left_out(self):
        fused = fuse_data(_frame(100, "a"), _frame(10, "b"), fs=100)
        self.assertEqual(list(fused.columns), ["T", "time", "a", "b"])

    def test_text_columns_left_out_when_both_interpolated(self):
        fused = fuse({"m1": _frame(100, "a"), "m2": _frame(10, "b")}, ["m1", "m2"], 100, prefix=True)
        self.assertEqual(list(fused.columns), ["T", "m1.a", "m2.b"])
        np.testing.assert_allclose(fused["m2.b"].dropna(), fused["T"][fused["m2.b"].notna()] * 2)


if __name__ == "__main__":
    unittest.main()
