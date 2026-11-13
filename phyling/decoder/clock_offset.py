"""Mini/Nano BLE clock offset correction, shared by the file decoder and the realtime manager.

A Mini/Nano relayed over BLE (by a Maxi or the mobile app) stamps its samples with its own clock. Each sample
carries notifDiff (us), the gap between the sample time and its reception by the relay: the smallest gap of a
continuous chunk, minus the mean BLE latency, is the clock offset to add to T.

Rule, per chunk (T gap > CHUNK_GAP_SEC splits chunks): no correction if the chunk lasts MIN_CHUNK_DURATION_SEC or
less, if every notifDiff is 0 (old Mini firmware) or if the offset exceeds MAX_SHIFT_SEC. notifDiff below
-MAX_SHIFT_SEC is an outlier.
"""

import logging

import numpy as np

MIN_CHUNK_DURATION_SEC = 15
MAX_SHIFT_SEC = 3
CHUNK_GAP_SEC = 3
NOTIF_LATENCY_SEC = 0.003  # mean time between a notification sent by the Mini and received by its relay
OUTLIER_NOTIF_DIFF_US = -MAX_SHIFT_SEC * 1e6

OFFSET_APPLIED = "applied"
OFFSET_TOO_SHORT = "too_short"
OFFSET_OLD_MINI = "old_mini"
OFFSET_TOO_LARGE = "too_large"


def chunk_offset(min_notif_diff_us: float, max_notif_diff_us: float, duration: float) -> tuple[float, str]:
    """Compute the clock offset of one chunk from its notifDiff bounds and duration.

    :param min_notif_diff_us: smallest notifDiff of the chunk (us), outliers excluded
    :param max_notif_diff_us: largest notifDiff of the chunk (us)
    :param duration: chunk duration (s)
    :return: (offset in s, status); the offset is to be applied only when the status is OFFSET_APPLIED
        (OFFSET_TOO_LARGE returns the rejected offset, for the log)
    """
    if duration <= MIN_CHUNK_DURATION_SEC:
        return 0.0, OFFSET_TOO_SHORT
    if min_notif_diff_us == 0 and max_notif_diff_us == 0:
        return 0.0, OFFSET_OLD_MINI
    offset = min_notif_diff_us / 1e6 - NOTIF_LATENCY_SEC
    if abs(offset) > MAX_SHIFT_SEC:
        return offset, OFFSET_TOO_LARGE
    return offset, OFFSET_APPLIED


def replace_outliers(notif_diff: np.ndarray) -> np.ndarray:
    """Replace in place the notifDiff outliers of a whole module by its largest value, so they never set the min."""
    outliers = notif_diff < OUTLIER_NOTIF_DIFF_US
    if outliers.any():
        notif_diff[outliers] = max(notif_diff)
    return notif_diff


def chunk_bounds(T: np.ndarray) -> list[int]:
    """Return the chunk boundaries of a time column: [0, <index after each gap > CHUNK_GAP_SEC>, len(T)]."""
    return [0, *list(np.where(np.diff(T) > CHUNK_GAP_SEC)[0] + 1), len(T)]


class ClockOffsetTracker:
    """Running clock offset of one realtime module stream: the file rule applied to the chunk received so far.

    Outliers are ignored (they never lower the min). A forward T gap > CHUNK_GAP_SEC starts a new chunk.
    """

    __slots__ = ("label", "start_t", "last_t", "min_us", "max_us", "offset", "status")

    def __init__(self, label: str = ""):
        self.label = label
        self.reset()

    def reset(self) -> None:
        """Forget the current chunk: the offset is 0 until the next chunk qualifies."""
        self.start_t = None
        self.last_t = None
        self.min_us = None
        self.max_us = None
        self.offset = 0.0
        self.status = OFFSET_TOO_SHORT

    def update(self, t: float, notif_diff_us: float | None) -> float:
        """Account one sample (raw T in s, notifDiff in us) and return the offset to add to its T and epoch."""
        if self.last_t is not None and t - self.last_t > CHUNK_GAP_SEC:
            self.reset()
        if self.start_t is None:
            self.start_t = self.last_t = t
        elif t > self.last_t:
            self.last_t = t
        if notif_diff_us is not None and notif_diff_us >= OUTLIER_NOTIF_DIFF_US:
            if self.min_us is None:
                self.min_us = self.max_us = notif_diff_us
            elif notif_diff_us < self.min_us:
                self.min_us = notif_diff_us
            elif notif_diff_us > self.max_us:
                self.max_us = notif_diff_us
        if self.min_us is None:
            return 0.0
        offset, status = chunk_offset(self.min_us, self.max_us, self.last_t - self.start_t)
        if status != self.status:
            self._log_status(offset, status)
            self.status = status
        self.offset = offset if status == OFFSET_APPLIED else 0.0
        return self.offset

    def _log_status(self, offset: float, status: str) -> None:
        """Log a change of correction status of the current chunk."""
        if status == OFFSET_APPLIED:
            logging.info(f"{self.label}: clock offset {offset * 1000:4.1f}ms applied (chunk T0 {self.start_t})")
        elif status == OFFSET_TOO_LARGE:
            logging.warning(
                f"{self.label}: realtime clock offset {offset * 1000:4.1f}ms is too large "
                f"(> {MAX_SHIFT_SEC * 1000:4.1f}ms), no correction applied"
            )
        elif status == OFFSET_OLD_MINI:
            logging.warning(f"{self.label}: cannot apply realtime clock offset to old miniphyling data")
