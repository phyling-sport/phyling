import asyncio
import signal
import sys
import time
from typing import Any
from typing import Union

import ujson
from bleak import BleakClient
from bleak import BleakScanner
from pandas import DataFrame

# UUIDs of characteristics
BLE_UUID_VERSION = "99064a28-8bef-4a2c-afe3-f17a28ebc8c3"
BLE_UUID_MAXI_VERSION = "5e143caa-f57b-440f-b59e-bf2fcfa1b838"
BLE_UUID_DATA = "00002A37-0000-1000-8000-00805F9B34FB"

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

ACC_FACTOR = 0.061 * (ACC_SCALE_SELECTION >> 1) / 1000 * 9.81
GYRO_FACTOR = 4.375 * (GYRO_RANGE / 125.0) / 1000.0
TEMP_FACTOR = 1.0 / 100


def find_device(name: str) -> Union[str, None]:
    """
    Find the BLE device by name. This function scans for available BLE devices and returns the address of the device
    with the specified name.

    :param name: Name of the BLE device to find

    :return: Address of the BLE device if found, None otherwise
    """
    print("Searching for BLE sensor...")
    devices = asyncio.run(BleakScanner.discover())
    for device in devices:
        if device.name == name:
            print(f"Sensor found: {name} ({device.address})")
            return device.address
    print("Sensor not found")
    return None


class NanoPhyling:
    name = None
    address = None
    config = {}
    disconnect = False
    df = None
    nbDatas = 0
    startBLETime = 0
    startRecordTime = 0

    def __init__(
        self,
        name: Union[str, None] = None,
        address: Union[str, None] = None,
        config: dict[str, Any] = NANO_DEF_CONFIG,
    ):
        """
        Initialize the NanoPhyling class. You have to provide the name OR the address of the BLE device.

        :param name: Name of the BLE device (default: None)
        :param address: Address of the BLE device (default: None)
        :param config: Configuration dictionary (default: NANO_DEF_CONFIG)
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

    def _notification_handler(self, sender, data):
        """
        Handle notifications from the BLE device. This function is called when data is received from the device.
        It decodes the data and appends it to the DataFrame.
        """
        T = int.from_bytes(data[:8], byteorder="little") / 1e6
        if self.startBLETime == 0:  # on first notification
            self.startBLETime = T  # set start time of BLE (to start df at 0)
            self.startRecordTime = (
                time.time()
            )  # set start time of record (to stop recording after duration)
        T = T - self.startBLETime
        deltaT = int.from_bytes(data[8:10], byteorder="little") / 1e6
        current_index = 10
        idx = 0
        # Get all datas
        while len(self.config["data"]) * 2 + current_index < len(data):
            line = []
            line.append(T + idx * deltaT)
            for i, value in enumerate(self.config["data"]):
                start_index = current_index + (i * 2)
                factor = 1
                if value.startswith("acc_"):
                    factor = ACC_FACTOR
                elif value.startswith("gyro_"):
                    factor = GYRO_FACTOR
                elif value.startswith("temp_"):
                    factor = TEMP_FACTOR
                line.append(
                    int.from_bytes(
                        data[start_index : start_index + 2],
                        byteorder="little",
                        signed=True,
                    )
                    * factor
                )
            current_index += len(self.config["data"]) * 2
            idx += 1
            self.nbDatas += 1
            self.df.loc[len(self.df)] = line
            if self.nbDatas % 500 == 0:
                print(f"Number of recorded data: {self.nbDatas}")

    def _signal_handler(self, sig, frame):
        """
        Handle signals (SIGINT, SIGTERM) to stop recording gracefully.
        """
        print("Interruption detected. Stopping recording...")
        self.disconnect = True

    async def _run_ble_client(self, duration: Union[int, None]):
        """
        Run the BLE client to connect to the device and start recording data.
        This function handles the connection, configuration, and data recording.

        :param duration: Duration in seconds to record data (default: None, record indefinitely, kill to stop)
        """
        self.df = DataFrame(columns=["T"] + self.config["data"])
        self.nbDatas = 0
        self.startRecordTime = 0
        async with BleakClient(self.address) as client:
            print("Connected to BLE sensor")

            # Send maxi version
            await client.write_gatt_char(
                BLE_UUID_MAXI_VERSION, b"v6.6.4", response=True
            )
            print("Maxi version sent")

            # Send configuration
            await client.write_gatt_char(
                BLE_UUID_DATA,
                BLE_NOTIF_NANO_SEND_CONFIG + bytes(ujson.dumps(self.config), "utf-8"),
            )
            print(f"Configuration sent: {self.config}")

            version = await client.read_gatt_char(BLE_UUID_VERSION)
            print(f"Sensor version: {version.decode('utf-8')}")

            # Enable notifications on the data characteristic
            await client.start_notify(BLE_UUID_DATA, self._notification_handler)
            print("Notifications enabled")

            # Start recording
            await client.write_gatt_char(BLE_UUID_DATA, BLE_NOTIF_START_REC)
            print("Recording started")

            # Wait for notifications until interruption
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            while not self.disconnect:
                if (
                    duration is not None
                    and self.startRecordTime > 0
                    and time.time() - self.startRecordTime > duration
                ):
                    self.disconnect = True
                    print(f"Stopping recording after {duration} seconds")
                    break
                else:
                    await asyncio.sleep(1)

            # Stop recording before exiting
            await client.write_gatt_char(BLE_UUID_DATA, BLE_NOTIF_STOP_REC)
            print("Recording stopped")
            await client.stop_notify(BLE_UUID_DATA)
            print("Notifications stopped")
            print(
                f"Number of recorded data: {self.nbDatas}, use NanoPhyling.get_df() to retrieve them"
            )

    def run(self, duration: Union[int, None] = None) -> None:
        """
        Run the BLE client to connect to the device and start recording data.
        This function handles the connection, configuration, and data recording.

        :param duration: Duration in seconds to record data (default: None, record indefinitely, kill to stop)
        """
        self.disconnect = False
        if not self.address:
            self.address = find_device(name=self.name)
        if not self.address:
            print(
                "Unable to find BLE sensor. Make sure it is turned on and within range."
            )
            return
        asyncio.run(self._run_ble_client(duration))

    def get_df(self) -> DataFrame:
        """
        Get the DataFrame containing the recorded data.
        :return: DataFrame with recorded data
        """
        return self.df


if __name__ == "__main__":
    nano = NanoPhyling(name=sys.argv[1])
    nano.run()
