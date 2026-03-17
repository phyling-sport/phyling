import asyncio
import struct
import time
from abc import ABC
from abc import abstractmethod
from typing import Union

import pandas as pd
from bleak import BleakClient
from bleak import BleakScanner
from pandas import DataFrame

from phyling import phyling_utils

# UUIDs of characteristics
BLE_UUID_INFOS = "74c3ca8b-f084-497c-9df9-7c7e1a69cd43"
BLE_UUID_VERSION = "99064a28-8bef-4a2c-afe3-f17a28ebc8c3"
BLE_UUID_MAXI_VERSION = "5e143caa-f57b-440f-b59e-bf2fcfa1b838"
BLE_UUID_PHYLING = "65a0c51d-eb0e-4f56-9346-c6925abb2bec"
BLE_UUID_PHYLING_OLD = "2a37"
BLE_UUID_CONFIG = "ddda4cc5-af99-4aa3-aaf7-819adbb9caeb"

# Values to write to start and stop recording
BLE_NOTIF_EMPTY = b"_"
BLE_NOTIF_STOP_REC = b"0"
BLE_NOTIF_START_REC = b"1"
BLE_NOTIF_TIME_SYNC = b"2"
BLE_NOTIF_TIME_OFFSET_SYNC = b"3"
BLE_NOTIF_NANO_SEND_CONFIG = b"4"

ACC_SCALE_SELECTION = 16
GYRO_RANGE = 2000
MAG_RANGE = 16

ACC_FACTOR = 0.061 * (ACC_SCALE_SELECTION >> 1) / 1000 * 9.81
GYRO_FACTOR = 4.375 * (GYRO_RANGE / 125.0) / 1000.0
MAG_FACTOR = (
    1.0 / 1711
    if MAG_RANGE == 16
    else (
        1.0 / 2281
        if MAG_RANGE == 12
        else (1.0 / 3421 if MAG_RANGE == 8 else (1.0 / 6842 if MAG_RANGE == 4 else 1.0))
    )
)
TEMP_FACTOR = 1.0 / 100

# empirically observed minimum of notifDiff across multiple recordings, used to correct the time
NOTIF_DIFF_OFFSET = 0.5

# Minimum connection duration (seconds) required to apply per-session time correction
MIN_CONN_TIME_SEC = 15

# BLE type character → (byte_size, signed, is_float)
TYPE_MAP = {
    "A": (1, True, False),  # int8
    "a": (1, False, False),  # uint8
    "B": (2, True, False),  # int16
    "b": (2, False, False),  # uint16
    "C": (4, True, False),  # int32
    "c": (4, False, False),  # uint32
    "D": (8, True, False),  # int64
    "d": (8, False, False),  # uint64
    "E": (4, False, True),  # float32
    "F": (8, False, True),  # float64
}


def find_device(name: str) -> Union[str, None]:
    """
    Scan BLE devices and return the address of the one matching name.

    :param name: BLE device name
    :return: Address if found, None otherwise
    """
    print(f"Scanning for BLE sensor '{name}'...")
    devices = asyncio.run(BleakScanner.discover())
    for device in devices:
        if device.name == name:
            print(f"Found: {name} ({device.address})")
            return device.address
    print(f"Device not found: {name}")
    return None


def _make_col_spec(name: str, type_char: str = "B") -> dict:
    """
    Build a column spec dict from a column name and BLE type character.

    NanoPhyling calls with type_char="B" (int16) for all columns.
    MiniPhyling calls with the actual type_char parsed from the device config.

    :param name: Column name (e.g. "acc_x", "gyro_z")
    :param type_char: BLE type character (see TYPE_MAP)
    :return: Dict with keys: name, size, signed, is_float, factor
    """
    size, signed, is_float = TYPE_MAP.get(type_char, (2, True, False))
    factor = 1.0
    if name.startswith("acc_"):
        factor = ACC_FACTOR
    elif name.startswith("gyro_"):
        factor = GYRO_FACTOR
    elif name.startswith("mag_"):
        factor = MAG_FACTOR
    elif name.startswith("temp_"):
        factor = TEMP_FACTOR
    return {
        "name": name,
        "size": size,
        "signed": signed,
        "is_float": is_float,
        "factor": factor,
    }


class BaseDevice(ABC):

    def __init__(
        self,
        ble_name: Union[str, None] = None,
        address: Union[str, None] = None,
        module_name: Union[str, None] = None,
    ):
        """
        Base class for BLE Phyling devices.

        :param ble_name: BLE device name (e.g. "NanoPhyling_42")
        :param address: BLE device address (alternative to ble_name)
        :param module_name: Custom module name (overrides auto-generated name in logs and df prefixes)
        """
        self.ble_name = ble_name
        self.address = address
        if self.ble_name is None and self.address is None:
            print("You must provide the ble_name or the address of the BLE device.")
        self._module_name = module_name
        self.calibration: dict = {}
        self.config = {}
        self._col_specs = []
        self._oneDataSize = 0
        self.df = None
        self.nbDatas = 0
        self.disconnect = False
        self._conn_id = 0
        self.startBLETime = 0
        self.startPCtime = 0
        self._ble_disconnected = False

    def get_name(self) -> str:
        """
        Return the module name for this device.
        - If module_name was set, returns it.
        - Otherwise, falls back to the BLE name.
        """
        return self._module_name or self.ble_name or self.address or "unknown"

    @abstractmethod
    async def _setup_config(self, client: BleakClient) -> None:
        """
        Device-specific setup called on each BLE connection.

        Must:
        1. Populate self.config["data"] and self._col_specs
        2. Call self._init_df_if_needed() to create the df on the first connection

        On reconnection: must NOT reinitialize the df (data continuity).
        """
        ...

    def set_module_name(self, module_name: str) -> None:
        """
        Set a custom module name for this device (used in logs and df prefixes).

        :param module_name: Custom name to set
        """
        self._module_name = module_name

    def get_df(self) -> Union[DataFrame, None]:
        """Return the recorded DataFrame with calibration applied."""
        if self.df is None:
            return None
        df = self.df.copy()
        for col, cal in self.calibration.items():
            if col in df.columns:
                coef = cal.get("coef", 1)
                offset = cal.get("offset", 0)
                df[col] = coef * (df[col] + offset)
        return df

    def get_calib(self) -> dict:
        """Return the current calibration dict."""
        return self.calibration

    def reset_calib(self) -> None:
        """Reset the calibration dict to empty (no calibration)."""
        self.calibration = {}

    def set_calib(self, calib: dict) -> None:
        """
        Replace the calibration dict.

        :param calib: Dict of {column: {"coef": float, "offset": float}}
        """
        self.calibration = calib

    def update_calib(self, calib: dict) -> None:
        """
        Merge a partial calibration dict into the current one.

        :param calib: Partial dict of {column: {"coef": float, "offset": float}}
        """
        self.calibration = phyling_utils.deep_merge(self.calibration, calib)

    def _init_df_if_needed(self) -> None:
        """Initialize self.df if not already done. Called from _setup_config."""
        if self.df is None:
            self.df = DataFrame(
                columns=["timestamp_nano", "notifDiff", "conn_id"] + self.config["data"]
            )

    def _notification_handler(self, _sender, data: bytes) -> None:
        """
        Handle BLE notifications. Shared across all device types.
        Uses self._col_specs to decode samples of variable type and size.
        """
        pc_now = time.time()

        current_index = 10
        nbPackets = int((len(data) - current_index) / self._oneDataSize)

        curBleTime = int.from_bytes(data[:8], byteorder="little") / 1e6
        elemSpacing = int.from_bytes(data[8:10], byteorder="little") / 1e6

        if self.startBLETime == 0:
            self.startBLETime = curBleTime
            self.startPCtime = pc_now

        timestamp_nano_start = self.startPCtime + curBleTime - self.startBLETime
        diffTimeNotif = pc_now - timestamp_nano_start + NOTIF_DIFF_OFFSET

        for packetIdx in range(nbPackets):
            timestamp_nano = (
                timestamp_nano_start
                + (elemSpacing * packetIdx)
                - (elemSpacing * (nbPackets - 1))
            )
            line = [timestamp_nano, diffTimeNotif, self._conn_id]

            offset = current_index
            for spec in self._col_specs:
                raw = data[offset : offset + spec["size"]]
                if spec["is_float"]:
                    fmt = "<f" if spec["size"] == 4 else "<d"
                    raw_val = struct.unpack(fmt, raw)[0]
                else:
                    raw_val = int.from_bytes(
                        raw, byteorder="little", signed=spec["signed"]
                    )
                val = raw_val * spec["factor"]
                line.append(val)
                offset += spec["size"]

            self.df.loc[len(self.df)] = line
            self.nbDatas += 1
            current_index += self._oneDataSize

    async def _run_ble_client(self, duration: Union[int, None]) -> None:
        """
        Run the BLE client to connect to the device and start recording data.
        Automatically reconnects on unexpected disconnection.
        Compatible with asyncio.gather for parallel execution.

        :param duration: Duration in seconds to record data (None = record indefinitely)
        """
        self.df = None  # Will be initialized in _setup_config on first connect
        self.nbDatas = 0
        self.startBLETime = 0
        self.startPCtime = 0
        self.disconnect = False
        self._conn_id = 0
        self._ble_disconnected = False

        start_time = time.time()

        while not self.disconnect:
            if duration is not None and time.time() - start_time >= duration:
                break

            self._ble_disconnected = False
            self.startBLETime = 0
            self.startPCtime = 0

            try:
                async with BleakClient(
                    self.address,
                    disconnected_callback=lambda __: setattr(
                        self, "_ble_disconnected", True
                    ),
                ) as client:
                    print(f"[{self.get_name()}] Device {self.ble_name} Connected")

                    await self._setup_config(client)

                    # Detect which data characteristic the device supports
                    all_char_uuids = [
                        str(c.uuid) for s in client.services for c in s.characteristics
                    ]
                    if BLE_UUID_PHYLING in all_char_uuids:
                        active_uuid = BLE_UUID_PHYLING
                    else:
                        active_uuid = BLE_UUID_PHYLING_OLD
                        print(f"[{self.get_name()}] Using legacy BLE UUID (old device)")

                    await client.start_notify(active_uuid, self._notification_handler)
                    try:
                        await client.write_gatt_char(active_uuid, BLE_NOTIF_START_REC)
                    except Exception:
                        pass  # Old devices may not support write on this characteristic
                    print(f"[{self.get_name()}] Recording started")

                    while not self.disconnect and not self._ble_disconnected:
                        if (
                            duration is not None
                            and time.time() - start_time >= duration
                        ):
                            self.disconnect = True
                            break
                        await asyncio.sleep(0.1)

                    if not self._ble_disconnected:
                        try:
                            await client.write_gatt_char(
                                active_uuid, BLE_NOTIF_STOP_REC
                            )
                            await client.stop_notify(active_uuid)
                        except Exception:
                            pass

                print(f"[{self.get_name()}] Recording stopped ({self.nbDatas} samples)")

                if self._ble_disconnected and not self.disconnect:
                    print(
                        f"[{self.get_name()}] Unexpected disconnect. Reconnecting in 1s..."
                    )
                    self._conn_id += 1
                    await asyncio.sleep(1.0)

            except (asyncio.CancelledError, KeyboardInterrupt):
                self.disconnect = True
            except Exception as e:
                if self.disconnect:
                    break
                print(
                    f"[{self.get_name()}] Connection error: {e}. Reconnecting in 1s..."
                )
                self._conn_id += 1
                await asyncio.sleep(1.0)

    def run(self, duration: Union[int, None] = None) -> None:
        """
        Connect to the device and start recording data.

        :param duration: Duration in seconds (None = record indefinitely)
        """
        self.disconnect = False
        if not self.address:
            self.address = find_device(name=self.ble_name)
        if not self.address:
            print(
                f"[{self.get_name()}] Device not found. Make sure it is turned on and within range."
            )
            return
        try:
            asyncio.run(self._run_ble_client(duration))
        except KeyboardInterrupt:
            print(f"[{self.get_name()}] Recording interrupted.")
        finally:
            self._apply_sync()

    async def _tare_async(self, columns: list) -> None:
        """
        Async version of tare() for use with asyncio.gather in PhylingDevices.
        Records 5 seconds of data to compute per-column mean offsets.
        Does NOT overwrite the existing df.
        """
        print(f"[{self.get_name()}] Tare: do not touch the device for 5 seconds...")
        saved_df = self.df
        saved_nbDatas = self.nbDatas
        self.disconnect = False
        await self._run_ble_client(5)
        tare_df = self.df
        self.df = saved_df
        self.nbDatas = saved_nbDatas
        if tare_df is None or len(tare_df) == 0:
            print(f"[{self.get_name()}] Tare failed: no data collected.")
            return
        for col in columns:
            if col in tare_df.columns:
                mean_val = float(tare_df[col].mean())
                coef = self.calibration.get(col, {}).get("coef", 1)
                self.calibration[col] = {"coef": coef, "offset": -mean_val}
                print(f"[{self.get_name()}] Tare {col}: offset={-mean_val:.6f}")
            else:
                print(f"[{self.get_name()}] Tare warning: column '{col}' not found.")

    def tare(self, columns: list) -> None:
        """
        Zero-center the given columns by recording 5 seconds of stationary data.
        Stores coef/offset in self.calibration, applied at get_df() time.
        The existing df is preserved.

        :param columns: List of column names to tare (e.g. ["gyro_z", "mag_x"])
        """
        self.disconnect = False
        if not self.address:
            self.address = find_device(name=self.ble_name)
        if not self.address:
            print(f"[{self.get_name()}] Device not found.")
            return
        asyncio.run(self._tare_async(columns))

    def _apply_sync(self, t0: float = None) -> None:
        """
        Apply time synchronization to the recorded DataFrame.
        Corrects device clock drift per connection session, computes T relative to t0
        (or start of recording), adds ISO 8601 'time' column, 'fs' column, and removes
        raw timestamp columns.

        :param t0: Common reference time (Unix timestamp). If None, uses first timestamp_nano.
        """
        if self.df is None or len(self.df) == 0:
            return

        # 1. Compute fs from timestamp_nano and add as a column
        self.df["fs"] = 1 / self.df["timestamp_nano"].diff()

        # 2. Per-connection time correction (one offset per conn_id session)
        for conn_id, group in self.df.groupby("conn_id"):
            mask = self.df["conn_id"] == conn_id
            duration = (
                group["timestamp_nano"].iloc[-1] - group["timestamp_nano"].iloc[0]
            )
            if duration > MIN_CONN_TIME_SEC:
                offset = (
                    group["notifDiff"].min() - NOTIF_DIFF_OFFSET - 0.003
                )  # remove 3ms
            else:
                offset = 0.0
            self.df.loc[mask, "timestamp_nano"] += offset

        # 3. Compute T relative to t0 (common reference) or start of this recording
        ref = t0 if t0 is not None else self.df["timestamp_nano"].iloc[0]
        self.df["T"] = self.df["timestamp_nano"] - ref

        # 4. Add ISO 8601 'time' column
        abs_ts = pd.to_datetime(self.df["timestamp_nano"], unit="s", utc=True)
        self.df["time"] = (
            abs_ts.dt.strftime("%Y-%m-%dT%H:%M:%S.")
            + abs_ts.dt.strftime("%f").str[:3]
            + "Z"
        )

        # 5. Remove raw timestamp columns, reorder
        data_cols = self.config["data"]
        self.df = self.df[["T", "time", "fs"] + data_cols].reset_index(drop=True)
