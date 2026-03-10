import asyncio
import time
from typing import Union

import pandas as pd
from bleak import BleakScanner

from phyling.phyling_utils import fuse_data


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
            if device.name in discovered_map:
                device.address = discovered_map[device.name]
                print(f"Found: {device.name} ({device.address})")
            else:
                print(f"Device not found: {device.name}")

    def calibrate_gyro(self) -> None:
        """
        Calibrate all devices simultaneously (parallel BLE connections).
        All devices must remain stationary for ~5 seconds.
        """
        self._scan_all()
        missing = [d.name for d in self.devices if not d.address]
        if missing:
            print(f"Cannot calibrate — devices not found: {missing}")
            return

        print(
            "Calibrating all devices simultaneously — do not touch any device for 5 seconds..."
        )
        asyncio.run(asyncio.gather(*(d._calibrate_gyro_async() for d in self.devices)))

    def run(self, duration: Union[int, None] = None) -> None:
        """
        Connect and record data from all devices simultaneously.
        After recording, time synchronization is applied to each device's DataFrame
        using a common T=0 reference captured just before starting.

        :param duration: Duration in seconds. None = record until interrupted.
        """
        self._scan_all()
        missing = [d.name for d in self.devices if not d.address]
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
            if device.name == name:
                return device
        raise ValueError(f"Device '{name}' not found in PhylingDevices.")
