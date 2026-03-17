import asyncio
import time
from typing import Union

import pandas as pd
from bleak import BleakScanner

from phyling.decoder.decoder import fuse_data


class PhylingDevices:
    def __init__(
        self, devices: list, output_fs: Union[float, None] = None, verbose: bool = False
    ):
        """
        Manage multiple BLE Phyling devices (NanoPhyling, MiniPhyling, etc.) simultaneously.

        :param devices: List of device instances (NanoPhyling, etc.)
        :param output_fs: Output sampling frequency for the fused DataFrame.
                          Defaults to the maximum rate among all devices.
        :param verbose: If True, print extra debug information.
        """
        self.devices = devices
        self.output_fs = output_fs or max(d.config["rate"] for d in devices)
        self.verbose = verbose

    def _scan_all(self) -> None:
        """
        Perform a single BLE scan to resolve addresses for all devices that don't have one.
        """
        needs_scan = [d for d in self.devices if not d.address]
        if not needs_scan:
            return

        print("Scanning for BLE devices...")
        discovered = asyncio.run(BleakScanner.discover())
        discovered_map = {dev.name: dev.address for dev in discovered if dev.name}

        for device in needs_scan:
            if device.ble_name in discovered_map:
                device.address = discovered_map[device.ble_name]
                print(f"Found: {device.ble_name} ({device.address})")
            else:
                print(f"Device not found: {device.ble_name}")

    def get_calib(self) -> dict:
        """Return the full calibration dict indexed by device name."""
        return {d.get_name(): d.get_calib() for d in self.devices}

    def reset_calib(self) -> None:
        """Reset calibration to default for all devices."""
        for device in self.devices:
            device.reset_calib()

    def set_calib(self, calib: dict) -> None:
        """
        Set calibration for each device by name.

        :param calib: Dict of {device_name: {column: {"coef": float, "offset": float}}}
        """
        for device in self.devices:
            if device.get_name() in calib:
                device.set_calib(calib[device.get_name()])

    def update_calib(self, calib: dict) -> None:
        """
        Merge a partial calibration dict into each device's calibration.

        :param calib: Dict of {device_name: {column: {"coef": float, "offset": float}}}
        """
        for device in self.devices:
            if device.get_name() in calib:
                device.update_calib(calib[device.get_name()])

    async def _tare_async(self, tare_dict: dict) -> None:
        tasks = []
        for device in self.devices:
            columns = tare_dict.get(device.get_name(), [])
            if columns:
                tasks.append(device._tare_async(columns))
        if tasks:
            await asyncio.gather(*tasks)

    def tare(self, tare_dict: dict) -> None:
        """
        Zero-center columns on specific devices by recording 5 seconds of stationary data.
        Runs all devices in parallel. The existing df of each device is preserved.

        :param tare_dict: Dict of {device_name: [col1, col2, ...]}
                          e.g. {"mini-42": ["adc_0", "mag_z"], "nano-07": ["gyro_z"]}
        """
        self._scan_all()
        missing = [
            d.ble_name
            for d in self.devices
            if d.get_name() in tare_dict and not d.address
        ]
        if missing:
            print(f"Cannot tare — devices not found: {missing}")
            return
        asyncio.run(self._tare_async(tare_dict))

    def run(self, duration: Union[int, None] = None) -> None:
        """
        Connect and record data from all devices simultaneously.
        After recording, time synchronization is applied to each device's DataFrame
        using a common T=0 reference captured just before starting.

        :param duration: Duration in seconds. None = record until interrupted.
        """
        self._scan_all()
        missing = [d.ble_name for d in self.devices if not d.address]
        if missing:
            print(f"Cannot run — devices not found: {missing}")
            return

        # Capture common reference time just before launching all devices
        t0 = time.time()
        try:
            asyncio.run(
                asyncio.gather(*(d._run_ble_client(duration) for d in self.devices))
            )
        except KeyboardInterrupt:
            print("Recording interrupted.")
        finally:
            # Apply sync with common t0 so all devices share T=0
            for device in self.devices:
                device._apply_sync(t0=t0)

    def get_df(self, drop_nan: bool = False) -> pd.DataFrame:
        """
        Return a single fused DataFrame with all devices' data aligned on a common time axis.
        Columns are prefixed by device name (e.g. NanoPhyling_42.acc_x).
        Columns: T, time, <device_name>.<col>, ...

        By default, the time axis spans from the earliest to the latest sample across all devices.
        Missing data at the edges of a device's recording are filled with NaN.

        :param drop_nan: If True, keep only rows where all devices have data (intersection).
                         If False (default), return the full union range with NaN for missing edges.
        :return: Fused DataFrame
        """
        dfs = [d.get_df() for d in self.devices]
        names = [d.get_name() for d in self.devices]

        if any(df is None or len(df) == 0 for df in dfs):
            raise ValueError("Some devices have no data. Run devices.run() first.")

        # Exclude metadata columns from fuse_data (handled separately)
        def data_cols(df):
            return [c for c in df.columns if c not in ("T", "time")]

        if len(self.devices) == 1:
            fused = dfs[0].copy()
            data_columns = data_cols(fused)
            fused = fused.rename(columns={c: f"{names[0]}.{c}" for c in data_columns})
            if drop_nan:
                fused = fused.dropna()
            return fused.reset_index(drop=True)

        fused = fuse_data(
            dfs[0],
            dfs[1],
            cols1=data_cols(dfs[0]),
            cols2=data_cols(dfs[1]),
            prefix1=f"{names[0]}.",
            prefix2=f"{names[1]}.",
            fs=self.output_fs,
            interp_df1=True,
            type_="union",
        )

        for i in range(2, len(self.devices)):
            fused = fuse_data(
                fused,
                dfs[i],
                cols2=data_cols(dfs[i]),
                prefix2=f"{names[i]}.",
                fs=self.output_fs,
                type_="union",
            )

        if drop_nan:
            fused = fused.dropna()

        # Recompute 'time' column from fused T using the first device's time as anchor
        first_time = pd.to_datetime(
            dfs[0]["time"].iloc[0], format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True
        )
        t_offset = dfs[0]["T"].iloc[0]
        abs_times = first_time + pd.to_timedelta(fused["T"] - t_offset, unit="s")
        fused.insert(
            1,
            "time",
            abs_times.dt.strftime("%Y-%m-%dT%H:%M:%S.")
            + abs_times.dt.strftime("%f").str[:3]
            + "Z",
        )

        return fused.reset_index(drop=True)

    def get_device(self, name: str):
        """
        Retrieve a device instance by its name.

        :param name: Device name (e.g. "NanoPhyling_42")
        :return: Device instance
        """
        for device in self.devices:
            if device.get_name() == name:
                return device
        raise ValueError(f"Device '{name}' not found in PhylingDevices.")
