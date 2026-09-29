import struct

from packaging import version

from phyling.decoder.decoder_utils import TIME_MODULE_ID
from phyling.decoder.decoder_utils import TIME_MODULE_SIZE

SOCKET_HEADER_SIZE = 5
DEDUP_KEY_SIZE = 9  # [module_id u8][ts i64]
WINDOW_MARGIN = 2.0
# a module table is pruned once it holds 1.5x the frames its window can cover
PRUNE_FACTOR = 1.5
LINK_RAW = 1  # binary topic data/phyraw/all (Maxi, Phyling-LTE over WiFi)
LINK_B64 = 2  # base64url topic data/phyraw/all/b64 (Phyling-LTE over LTE)
LINK_NAMES = {LINK_RAW: "raw", LINK_B64: "b64"}

NO_TS = -(1 << 64)  # below any i64 timestamp: "nothing seen yet"

_TS_STRUCT = struct.Struct("<q")


def build_frame_sizes(header: dict) -> dict:
    """Build the {module_id: frame size} table of a recDescription.

    Same id resolution as decoder_utils.getModName: the time update module (id 100) exists from v6.6.0 on.

    Args:
        header (dict): The recDescription

    Returns:
        dict: frame size in bytes per module id
    """
    sizes = {mod["id"]: mod["size"] for mod in header["modules"].values()}
    if version.parse(header["description"].get("version", "v6.0.0")) >= version.parse(
        "v6.6.0"
    ):
        sizes[TIME_MODULE_ID] = TIME_MODULE_SIZE
    return sizes


def _module_rate(mod: dict) -> float:
    """Return the realtime rate (Hz) of a recDescription module, its nominal rate as fallback, 0 if unknown."""
    return mod.get("realtimeRate") or mod.get("rate") or 0


def build_window_us(header: dict, ring_bytes: int) -> int:
    """Return the dedup window of a recDescription, in µs (0 when no module has a rate).

    The window is the time the device ring holds at the recDescription realtime throughput
    (sum over the modules of rate x frame size), times WINDOW_MARGIN: a frame is never replayed later than that
    after the newest one. It is shared by all modules, so the state stays bounded by the ring capacity in frames
    whatever the number of modules. A module without a positive rate is never deduplicated.

    Args:
        header (dict): The recDescription
        ring_bytes (int): Capacity of the device realtime ring, in bytes

    Returns:
        int: window in µs
    """
    throughput = sum(
        _module_rate(mod) * mod["size"]
        for mod in header["modules"].values()
        if _module_rate(mod) > 0
    )
    if throughput <= 0:
        return 0
    return int(WINDOW_MARGIN * ring_bytes / throughput * 1e6)


class _ModuleState:
    """Dedup state of one module: seen timestamps with the links that carried them, and watermarks."""

    __slots__ = (
        "window_us",
        "seen",
        "max_ts",
        "floor",
        "link_max_ts",
        "prune_at",
        "expected",
    )

    def __init__(self, window_us: int, expected: int) -> None:
        self.window_us = window_us
        self.expected = expected
        self.clear()

    def clear(self) -> None:
        """Forget every timestamp and watermark."""
        self.seen = {}  # ts -> bitmask of the links that carried the frame
        self.max_ts = NO_TS
        self.floor = NO_TS  # watermark: max_ts - window_us
        self.link_max_ts = [NO_TS, NO_TS, NO_TS]  # indexed by LINK_RAW / LINK_B64
        self.prune_at = PRUNE_FACTOR * self.expected


class RealtimeDedup:
    """Per device dedup of realtime frames, for one recDescription of one record.

    A realtime payload is `[5 bytes socket header][frame][frame]...`, each frame `[module_id u8][ts i64 µs][fields]`
    with a size fixed per module by the recDescription. The same frame can reach the server twice: a Phyling-LTE in
    dual mode publishes the same stream over LTE (".../b64" topic) and WiFi (binary topic) with a different chunking,
    and a device replays its own ring after an outage. Within one record no two distinct frames share their first 9
    bytes, so `[module_id][ts]` is the dedup key; the state must be dropped at every record boundary.

    Frames are split by their module size, keyed on their first 9 bytes, and a frame already seen is removed from
    the payload. A frame older than its module watermark (newest ts minus the window) always passes. Whatever cannot
    be split (unknown id, truncated frame) passes untouched, so the decoder keeps its own resync.

    Counters (cumulative since the last pop_counters): "dropped" duplicates, "below_watermark" frames passed
    without dedup, "unsplit" payloads with a remainder passed without dedup, and "single_link" per link name: frames
    evicted having been carried by one link only while the other link had streamed past them.
    """

    def __init__(self, header: dict, ring_bytes: int) -> None:
        """Build the dedup state of a recDescription.

        Args:
            header (dict): The recDescription
            ring_bytes (int): Capacity of the device realtime ring, in bytes
        """
        self.sizes = build_frame_sizes(header)
        self.modules = {}
        self.window_us = build_window_us(header, ring_bytes)
        for mod in header["modules"].values():
            rate = _module_rate(mod)
            if self.window_us > 0 and rate > 0 and mod["size"] >= DEDUP_KEY_SIZE:
                expected = int(rate * self.window_us / 1e6) + 64
                self.modules[mod["id"]] = _ModuleState(self.window_us, expected)
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.dropped = 0
        self.below_watermark = 0
        self.unsplit = 0
        self.single_link = {name: 0 for name in LINK_NAMES.values()}

    def pop_counters(self) -> dict:
        """Return the counters accumulated since the last call and reset them."""
        counters = {
            "dropped": self.dropped,
            "below_watermark": self.below_watermark,
            "unsplit": self.unsplit,
            "single_link": dict(self.single_link),
        }
        self._reset_counters()
        return counters

    def filter(self, payload: bytes, link: int = LINK_RAW) -> bytes | None:
        """Remove the already seen frames from a payload.

        Args:
            payload (bytes): The raw payload, socket header included
            link (int): LINK_RAW or LINK_B64, the topic the payload came from

        Returns:
            bytes | None: the payload itself if nothing was removed, a rebuilt `header + new frames + remainder`
            otherwise, None if nothing is left to decode
        """
        sizes_get = self.sizes.get
        modules_get = self.modules.get
        unpack_ts = _TS_STRUCT.unpack_from
        n = len(payload)
        pos = SOCKET_HEADER_SIZE
        dropped = below = 0
        out = None  # built lazily on the first duplicate, until then the payload is passed as is
        while pos < n:
            mod_id = payload[pos]
            size = sizes_get(mod_id)
            if size is None or pos + size > n:
                self.unsplit += 1
                break
            st = modules_get(mod_id)
            keep = True
            if st is not None:
                ts = unpack_ts(payload, pos + 1)[0]
                if ts < st.floor:
                    below += 1
                else:
                    seen = st.seen
                    mask = seen.get(ts)
                    if mask is None:
                        seen[ts] = link
                        if ts > st.max_ts:  # a new max is also the new max of its link
                            st.max_ts = ts
                            st.floor = ts - st.window_us
                            st.link_max_ts[link] = ts
                        elif ts > st.link_max_ts[link]:
                            st.link_max_ts[link] = ts
                        if len(seen) > st.prune_at:
                            self._prune(st)
                    else:
                        seen[ts] = mask | link
                        keep = False
                        if ts > st.link_max_ts[link]:
                            st.link_max_ts[link] = ts
            if keep:
                if out is not None:
                    out += payload[pos : pos + size]
            else:
                dropped += 1
                if out is None:
                    out = bytearray(payload[:pos])
            pos += size
        self.dropped += dropped
        self.below_watermark += below
        if out is None:
            return payload
        out += payload[pos:]
        if len(out) <= SOCKET_HEADER_SIZE:
            return None
        return bytes(out)

    def _count_single_link(self, st: _ModuleState, ts: int, mask: int) -> None:
        """Count an evicted frame carried by one link only, if the other link had streamed past it."""
        if mask == LINK_RAW or mask == LINK_B64:
            if st.link_max_ts[LINK_B64 if mask == LINK_RAW else LINK_RAW] >= ts:
                self.single_link[LINK_NAMES[mask]] += 1

    def _prune(self, st: _ModuleState) -> None:
        """Evict the timestamps under the module watermark (amortized: runs once the table grew by PRUNE_FACTOR)."""
        floor = st.floor
        kept = {}
        for ts, mask in st.seen.items():
            if ts >= floor:
                kept[ts] = mask
            else:
                self._count_single_link(st, ts, mask)
        st.seen = kept
        st.prune_at = PRUNE_FACTOR * max(st.expected, len(kept))

    def flush(self) -> None:
        """Evict the whole state (record boundary), accounting the frames seen on a single link."""
        for st in self.modules.values():
            for ts, mask in st.seen.items():
                self._count_single_link(st, ts, mask)
            st.clear()
