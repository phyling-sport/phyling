import asyncio
import re
import time
from typing import Any
from typing import Union

import pandas as pd
import ujson
from bleak import BleakClient
from bleak import BleakScanner
from pandas import DataFrame

# UUIDs of characteristics
BLE_UUID_VERSION = "99064a28-8bef-4a2c-afe3-f17a28ebc8c3"
BLE_UUID_MAXI_VERSION = "5e143caa-f57b-440f-b59e-bf2fcfa1b838"
BLE_UUID_PHYLING = "65a0c51d-eb0e-4f56-9346-c6925abb2bec"

# Values to write to start and stop recording
BLE_NOTIF_EMPTY = b"_"
BLE_NOTIF_STOP_REC = b"0"
BLE_NOTIF_START_REC = b"1"
BLE_NOTIF_TIME_SYNC = b"2"
BLE_NOTIF_TIME_OFFSET_SYNC = b"3"
BLE_NOTIF_NANO_SEND_CONFIG = b"4"

NANO_DEF_CONFIG = {
    "rate": 200,
    "bufferSize": 200,
    "data": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
}

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


class NanoPhyling:

    def __init__(
        self,
        name: Union[str, None] = None,
        address: Union[str, None] = None,
        config: dict[str, Any] = NANO_DEF_CONFIG,
        gyro_offsets: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0},
        display_name: Union[str, None] = None,
    ):
        """
        Initialize the NanoPhyling class. Provide name OR address of the BLE device.

        :param name: BLE device name (e.g. "NanoPhyling_42")
        :param address: BLE device address
        :param config: Configuration dictionary (default: NANO_DEF_CONFIG)
        :param gyro_offsets: Gyroscope offsets for calibration (default: all zeros)
        :param display_name: Custom display name (overrides auto-generated name in logs and df prefixes)
        """
        self.name = name
        self.address = address
        if self.name is None and self.address is None:
            print("You must provide the name or the address of the BLE device.")
            return
        self.config = config
        if "rate" not in self.config:
            self.config["rate"] = NANO_DEF_CONFIG["rate"]
        if "bufferSize" not in self.config:
            self.config["bufferSize"] = NANO_DEF_CONFIG["bufferSize"]
        if "data" not in self.config:
            self.config["data"] = NANO_DEF_CONFIG["data"]
        self.disconnect = False
        self.df = None
        self.nbDatas = 0
        self.startBLETime = 0
        self.startRecordTime = 0
        self._conn_id = 0
        self.gyro_offsets = gyro_offsets
        self._display_name = display_name

    def get_name(self) -> str:
        """
        Return the display name for this device.
        - If display_name was set, returns it.
        - Otherwise extracts the number from "NanoPhyling_XX" → "nano-XX".
        - Falls back to the BLE name if no number is found.
        """
        if self._display_name:
            return self._display_name
        if self.name:
            match = re.search(r"_(\d+)$", self.name)
            if match:
                return f"nano-{match.group(1)}"
            return self.name
        return self.address or "unknown"

    def set_display_name(self, display_name: str) -> None:
        """
        Set a custom display name for this device (used in logs and df prefixes).
        Overrides the default auto-generated name based on the BLE name.

        :param display_name: Custom name to set
        """
        self._display_name = display_name

    def _notification_handler(self, _sender, data):
        """
        Handle BLE notifications and generate:
          - T: temps relatif a l'horloge du nano (en secondes, float)
          - notifDiff: différence de temps entre le PC et le nano (en secondes, float)
        """
        # dt = 1.0 / self.config["rate"]
        pc_now = time.time()

        curBleTime = int.from_bytes(data[:8], byteorder="little") / 1e6
        elemSpacing = int.from_bytes(data[8:10], byteorder="little") / 1e6

        if self.startBLETime == 0:
            self.startBLETime = curBleTime
            self.startPCtime = pc_now

        current_index = 10
        num_sensors = len(self.config["data"])
        sampleIndex = 0
        timestamp_nano_start = self.startPCtime + curBleTime - self.startBLETime
        diffTimeNotif = pc_now - timestamp_nano_start + NOTIF_DIFF_OFFSET

        while current_index + 2 * num_sensors <= len(data):
            # t_rel = sampleIndex * dt
            timestamp_nano = timestamp_nano_start + (elemSpacing * sampleIndex)

            line = [timestamp_nano, diffTimeNotif, self._conn_id]

            for i, value_name in enumerate(self.config["data"]):
                start_index = current_index + i * 2
                raw_val = int.from_bytes(
                    data[start_index : start_index + 2],
                    byteorder="little",
                    signed=True,
                )

                factor = 1.0
                offset = 0.0
                if value_name.startswith("acc_"):
                    factor = ACC_FACTOR
                elif value_name.startswith("gyro_"):
                    factor = GYRO_FACTOR
                    axis = value_name.split("_")[1]
                    offset = self.gyro_offsets.get(axis, 0.0)
                elif value_name.startswith("temp_"):
                    factor = TEMP_FACTOR
                elif value_name.startswith("mag_"):
                    factor = MAG_FACTOR

                line.append((raw_val * factor) - offset)

            self.df.loc[len(self.df)] = line

            sampleIndex += 1
            self.nbDatas += 1
            current_index += 2 * num_sensors

    def _signal_handler(self, sig, frame):
        """Handle signals (SIGINT, SIGTERM) to stop recording gracefully."""
        print(f"[{self.get_name()}] Interruption detected. Stopping recording...")
        self.disconnect = True

    def _apply_sync(self, t0: float = None) -> None:
        """
        Apply time synchronization to the recorded DataFrame.
        Corrects nano clock drift per connection session, computes T relative to t0
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
                # print(
                #     f"[{self.get_name()}] Session {conn_id}: applied time offset "
                #     f"{offset * 1000:.1f}ms ({duration:.0f}s of data)"
                # )
            else:
                offset = 0.0
                # print(
                #     f"[{self.get_name()}] Session {conn_id}: connected only {duration:.1f}s, "
                #     f"no time offset applied (minimum {MIN_CONN_TIME_SEC}s required)"
                # )
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

    async def _run_ble_client(self, duration: Union[int, None]):
        """
        Run the BLE client to connect to the device and start recording data.
        Automatically reconnects on unexpected disconnection.
        Compatible with asyncio.gather for parallel execution.

        :param duration: Duration in seconds to record data (default: None, record indefinitely)
        """
        self.df = DataFrame(
            columns=["timestamp_nano", "notifDiff", "conn_id"] + self.config["data"]
        )
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
                    print(f"[{self.get_name()}] Device {self.name} Connected")

                    version = b"v7.0.3"
                    await client.write_gatt_char(
                        BLE_UUID_MAXI_VERSION, version, response=True
                    )

                    await client.write_gatt_char(
                        BLE_UUID_PHYLING,
                        BLE_NOTIF_NANO_SEND_CONFIG
                        + bytes(ujson.dumps(self.config), "utf-8"),
                    )

                    await client.read_gatt_char(BLE_UUID_VERSION)

                    await client.start_notify(
                        BLE_UUID_PHYLING, self._notification_handler
                    )

                    await client.write_gatt_char(BLE_UUID_PHYLING, BLE_NOTIF_START_REC)
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
                                BLE_UUID_PHYLING, BLE_NOTIF_STOP_REC
                            )
                            await client.stop_notify(BLE_UUID_PHYLING)
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
        Run the BLE client to connect to the device and start recording data.

        :param duration: Duration in seconds to record data (default: None, record indefinitely)
        """
        self.disconnect = False
        if not self.address:
            self.address = find_device(name=self.name)
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

    async def _calibrate_gyro_async(self) -> dict:
        """
        Async version of calibrate_gyro() for use with asyncio.gather in PhylingDevices.
        Calculates gyroscope bias by averaging 5 seconds of stationary data.
        """
        print(
            f"[{self.get_name()}] Calibration: do not touch the device for 5 seconds..."
        )

        old_gyro_offsets = self.gyro_offsets
        self.gyro_offsets = {"x": 0.0, "y": 0.0, "z": 0.0}

        await self._run_ble_client(5)

        try:
            self.gyro_offsets = {
                "x": float(self.df["gyro_x"].mean()),
                "y": float(self.df["gyro_y"].mean()),
                "z": float(self.df["gyro_z"].mean()),
            }
            print(f"[{self.get_name()}] Calibration done. Offsets: {self.gyro_offsets}")
        except KeyError:
            print(
                f"[{self.get_name()}] Calibration error: columns gyro_x/y/z not found."
            )
            self.gyro_offsets = old_gyro_offsets

        return self.gyro_offsets

    def calibrate_gyro(self) -> dict:
        """
        Calculates gyroscope bias by averaging 5 seconds of stationary data.
        Stores offsets in the instance to automatically zero-center future measurements.
        """
        self.disconnect = False
        if not self.address:
            self.address = find_device(name=self.name)

        return asyncio.run(self._calibrate_gyro_async())

    def get_df(self) -> DataFrame:
        """
        Get the DataFrame containing the recorded data.
        :return: DataFrame with columns: T, time, <data_cols>, fs
        """
        return self.df
