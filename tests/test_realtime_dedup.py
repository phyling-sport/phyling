import random
import struct
import unittest

from phyling.decoder.realtime_dedup import build_frame_sizes
from phyling.decoder.realtime_dedup import build_window_us
from phyling.decoder.realtime_dedup import LINK_B64
from phyling.decoder.realtime_dedup import LINK_RAW
from phyling.decoder.realtime_dedup import RealtimeDedup
from phyling.decoder.realtime_dedup import SOCKET_HEADER_SIZE

EPOCH_US = 1_790_000_000_000_000
IMU_ID, GPS_ID = 1, 2
IMU_SIZE, GPS_SIZE = 23, 33
LTE_RING_BYTES = 24 * 1536
MAXI_RING_BYTES = 50 * 4096
SOCKET_HEADER = b"\x01\x3c\x00\x05\x00"


def make_header(version: str = "v7.0.0", imu_rate: float = 100, gps_rate: float = 10) -> dict:
    """Build a recDescription with an IMU and a GPS module."""
    return {
        "description": {"version": version, "epochUs": EPOCH_US},
        "modules": {
            "imu": {"id": IMU_ID, "size": IMU_SIZE, "realtimeRate": imu_rate},
            "gps": {"id": GPS_ID, "size": GPS_SIZE, "realtimeRate": gps_rate},
        },
    }


def imu_frame(i: int) -> bytes:
    """Return the i-th IMU frame of a 100 Hz stream."""
    return struct.pack("<Bq7h", IMU_ID, EPOCH_US + i * 10_000, i, 2, 3, 4, 5, 6, 7)


def gps_frame(i: int) -> bytes:
    """Return the i-th GPS frame of a 10 Hz stream."""
    return struct.pack("<Bq6f", GPS_ID, EPOCH_US + i * 100_000, 45.0, 5.0, 3.0, 100.0, 1.2, float(i))


def payload(frames: list) -> bytes:
    """Build a realtime payload from frames."""
    return SOCKET_HEADER + b"".join(frames)


def frames_of(data: bytes | None) -> list:
    """Split a filtered payload back into its frames (test helper, known sizes)."""
    if data is None:
        return []
    assert data[:SOCKET_HEADER_SIZE] == SOCKET_HEADER
    sizes = {IMU_ID: IMU_SIZE, GPS_ID: GPS_SIZE}
    out, pos = [], SOCKET_HEADER_SIZE
    while pos < len(data):
        size = sizes[data[pos]]
        out.append(data[pos : pos + size])
        pos += size
    return out


def chunk(frames: list, rng: random.Random, max_frames: int) -> list:
    """Cut a frame stream into payloads of random length."""
    out, i = [], 0
    while i < len(frames):
        n = rng.randint(1, max_frames)
        out.append(payload(frames[i : i + n]))
        i += n
    return out


class TablesTest(unittest.TestCase):
    """Frame size and window tables built from the recDescription."""

    def test_sizes_from_modules(self):
        self.assertEqual(
            build_frame_sizes(make_header("v6.5.0")),
            {IMU_ID: IMU_SIZE, GPS_ID: GPS_SIZE},
        )

    def test_time_module_from_v6_6_0(self):
        self.assertEqual(build_frame_sizes(make_header("v6.6.0"))[100], 13)

    def test_window_from_ring_capacity_and_throughput(self):
        throughput = 100 * IMU_SIZE + 10 * GPS_SIZE  # bytes/s of the recDescription
        window = build_window_us(make_header(), LTE_RING_BYTES)
        self.assertEqual(window, int(2 * LTE_RING_BYTES / throughput * 1e6))
        self.assertAlmostEqual(window / 1e6, 28.03, places=1)
        self.assertAlmostEqual(build_window_us(make_header(), MAXI_RING_BYTES) / 1e6, 155.7, places=1)

    def test_state_bounded_by_the_ring_capacity_in_frames(self):
        dedup = RealtimeDedup(make_header(), LTE_RING_BYTES)
        in_window = sum(st.expected - 64 for st in dedup.modules.values())
        ring_frames = LTE_RING_BYTES * (100 + 10) / (100 * IMU_SIZE + 10 * GPS_SIZE)
        self.assertAlmostEqual(in_window / ring_frames, 2.0, places=2)

    def test_module_without_rate_is_not_deduplicated(self):
        dedup = RealtimeDedup(make_header(imu_rate=0), LTE_RING_BYTES)
        self.assertNotIn(IMU_ID, dedup.modules)
        data = payload([imu_frame(0)])
        dedup.filter(data)
        self.assertIs(dedup.filter(data), data)

    def test_no_rate_at_all_disables_dedup(self):
        self.assertEqual(build_window_us(make_header(imu_rate=0, gps_rate=0), LTE_RING_BYTES), 0)
        self.assertEqual(
            RealtimeDedup(make_header(imu_rate=0, gps_rate=0), LTE_RING_BYTES).modules,
            {},
        )


class DedupTest(unittest.TestCase):
    """Frame-level dedup of payloads."""

    def setUp(self):
        self.dedup = RealtimeDedup(make_header(), LTE_RING_BYTES)

    def test_unique_payload_is_passed_as_is(self):
        data = payload([imu_frame(i) for i in range(10)])
        self.assertIs(self.dedup.filter(data), data)

    def test_partial_duplicate(self):
        self.dedup.filter(payload([imu_frame(i) for i in range(10)]))
        out = self.dedup.filter(payload([imu_frame(i) for i in range(5, 15)]))
        self.assertEqual(frames_of(out), [imu_frame(i) for i in range(10, 15)])
        self.assertEqual(self.dedup.pop_counters()["dropped"], 5)

    def test_full_duplicate_returns_none(self):
        data = payload([imu_frame(i) for i in range(10)] + [gps_frame(0)])
        self.dedup.filter(data)
        self.assertIsNone(self.dedup.filter(data))

    def test_duplicate_inside_one_payload(self):
        out = self.dedup.filter(payload([imu_frame(0), imu_frame(1), imu_frame(0)]))
        self.assertEqual(frames_of(out), [imu_frame(0), imu_frame(1)])

    def test_same_ts_on_two_modules_is_not_a_duplicate(self):
        imu = struct.pack("<Bq7h", IMU_ID, EPOCH_US, 0, 0, 0, 0, 0, 0, 0)
        gps = struct.pack("<Bq6f", GPS_ID, EPOCH_US, 0, 0, 0, 0, 0, 0)
        self.assertEqual(frames_of(self.dedup.filter(payload([imu, gps]))), [imu, gps])

    def test_two_chunkings_with_overlap_yield_each_frame_once(self):
        rng = random.Random(42)
        stream = []
        for i in range(600):
            stream.append(imu_frame(i))
            if i % 10 == 0:
                stream.append(gps_frame(i // 10))
        lte = [(p, LINK_B64) for p in chunk(stream, rng, 60)]
        wifi = [(p, LINK_RAW) for p in chunk(stream[100:], rng, 7)]  # WiFi joins late, finer chunks
        arrivals = lte + wifi
        rng.shuffle(arrivals)
        received = []
        for data, link in arrivals:
            received.extend(frames_of(self.dedup.filter(data, link)))
        self.assertEqual(sorted(received), sorted(stream))
        self.assertEqual(self.dedup.pop_counters()["dropped"], len(stream) - 100)

    def test_ring_replay_of_a_single_device(self):
        """A Maxi replays its own ring newest-first on the binary topic: every replayed frame is dropped."""
        dedup = RealtimeDedup(make_header(), MAXI_RING_BYTES)
        chunks = [payload([imu_frame(i) for i in range(k, k + 20)]) for k in range(0, 400, 20)]
        for data in chunks:
            dedup.filter(data, LINK_RAW)
        for data in reversed(chunks[5:]):
            self.assertIsNone(dedup.filter(data, LINK_RAW))
        dedup.flush()
        self.assertEqual(dedup.pop_counters()["single_link"], {"raw": 0, "b64": 0})

    def test_flush_forgets_everything(self):
        data = payload([imu_frame(i) for i in range(10)])
        self.dedup.filter(data)
        self.dedup.flush()
        self.assertIs(self.dedup.filter(data), data)

    def test_truncated_last_frame_is_passed_untouched(self):
        self.dedup.filter(payload([imu_frame(0), imu_frame(1)]))
        truncated = payload([imu_frame(0), imu_frame(2), imu_frame(1)[:10]])
        out = self.dedup.filter(truncated)
        self.assertEqual(out, payload([imu_frame(2)]) + imu_frame(1)[:10])
        self.assertEqual(self.dedup.pop_counters()["unsplit"], 1)

    def test_unknown_id_stops_the_split(self):
        self.dedup.filter(payload([imu_frame(0), imu_frame(1)]))
        rest = b"\x07" + imu_frame(0) + imu_frame(1)  # the duplicates after the garbage byte pass
        out = self.dedup.filter(payload([imu_frame(0)]) + rest)
        self.assertEqual(out, SOCKET_HEADER + rest)
        self.assertEqual(self.dedup.pop_counters()["unsplit"], 1)

    def test_frame_below_watermark_passes(self):
        window_s = build_window_us(make_header(), LTE_RING_BYTES) / 1e6
        old = imu_frame(0)
        self.dedup.filter(payload([old]))
        recent = imu_frame(int(window_s * 100) + 50)
        self.dedup.filter(payload([recent]))
        self.assertEqual(self.dedup.filter(payload([old])), payload([old]))
        self.assertEqual(self.dedup.pop_counters()["below_watermark"], 1)

    def test_prune_keeps_the_table_bounded_and_the_window_exact(self):
        dedup = RealtimeDedup(make_header(), LTE_RING_BYTES)
        st = dedup.modules[IMU_ID]
        window_frames = st.window_us // 10_000
        n = 50 * (10 * st.expected // 50)
        for k in range(0, n, 50):
            dedup.filter(payload([imu_frame(i) for i in range(k, k + 50)]))
        self.assertLessEqual(len(st.seen), st.prune_at + 1)
        self.assertIsNone(dedup.filter(payload([imu_frame(n - window_frames + 1)])))

    def test_time_module_frames_are_split_but_never_deduplicated(self):
        dedup = RealtimeDedup(make_header("v6.6.0"), LTE_RING_BYTES)
        time_frame = struct.pack("<BIQ", 100, 1000, EPOCH_US)
        data = SOCKET_HEADER + time_frame + time_frame + imu_frame(0)
        self.assertIs(dedup.filter(data), data)


class LinkCountersTest(unittest.TestCase):
    """Frames seen on a single link of a dual Phyling-LTE (LTE /b64 vs WiFi binary topic)."""

    def test_frames_carried_by_one_link_only(self):
        dedup = RealtimeDedup(make_header(), LTE_RING_BYTES)
        dedup.filter(payload([imu_frame(i) for i in range(20)]), LINK_B64)
        dedup.filter(payload([imu_frame(i) for i in range(15)]), LINK_RAW)
        dedup.filter(payload([imu_frame(i) for i in range(20, 30)]), LINK_RAW)
        dedup.flush()
        # 15..19: LTE only while WiFi streamed past them; 20..29: WiFi only but LTE never reached them
        self.assertEqual(dedup.pop_counters()["single_link"], {"raw": 0, "b64": 5})

    def test_single_topic_device_never_counts(self):
        dedup = RealtimeDedup(make_header(), LTE_RING_BYTES)
        dedup.filter(payload([imu_frame(i) for i in range(20)]), LINK_B64)
        dedup.flush()
        self.assertEqual(dedup.pop_counters()["single_link"], {"raw": 0, "b64": 0})

    def test_single_link_counted_on_prune(self):
        dedup = RealtimeDedup(make_header(), LTE_RING_BYTES)
        st = dedup.modules[IMU_ID]
        dedup.filter(payload([imu_frame(0)]), LINK_RAW)
        n = int(2 * st.expected)
        for k in range(1, n, 50):
            frames = payload([imu_frame(i) for i in range(k, min(k + 50, n))])
            dedup.filter(frames, LINK_RAW)
            dedup.filter(frames, LINK_B64)
        self.assertEqual(dedup.pop_counters()["single_link"], {"raw": 1, "b64": 0})


if __name__ == "__main__":
    unittest.main()
