import re
from typing import Union

from bleak import BleakClient

from phyling.ble.base_device import _make_col_spec
from phyling.ble.base_device import BaseDevice
from phyling.ble.base_device import BLE_UUID_CONFIG
from phyling.ble.base_device import BLE_UUID_INFOS


def _validate_crc8(data: bytes) -> bool:
    """
    Validate CRC8 checksum (polynomial 0x07) of config bytes.
    The last byte of data is the expected CRC.

    :return: True if valid, False otherwise (prints a warning)
    """
    crc = 0
    for byte in data[:-1]:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    if crc != data[-1]:
        print(f"Warning: CRC8 mismatch (expected {data[-1]:#x}, got {crc:#x})")
        return False
    return True


class MiniPhyling(BaseDevice):

    def __init__(
        self,
        ble_name: Union[str, None],
        address: Union[str, None] = None,
        module_name: Union[str, None] = None,
    ):
        """
        BLE client for Mini-Phyling devices. Provide ble_name OR address.

        Unlike NanoPhyling, the configuration is read from the device on connection —
        it cannot be modified from the client side.

        :param ble_name: BLE device name (e.g. "MiniPhyling_01")
        :param address: BLE device address
        :param module_name: Custom module name (overrides auto-generated name in logs and df prefixes)
        """
        super().__init__(ble_name, address, module_name)
        self.config = {"rate": 200, "bufferSize": 0, "data": []}

    async def _setup_config(self, client: BleakClient) -> None:
        """
        Read config from the device characteristic and build _col_specs.
        Config is only read on the first connection; subsequent reconnections reuse it.

        Config format: "module{col|type;col|type}..." with CRC8 as last byte.
        Example: "imu{acc_x|B;acc_y|B;acc_z|B;gyro_x|B;gyro_y|B;gyro_z|B}mag{mag_x|B;mag_y|B;mag_z|B}"
        """
        if not self.config["data"]:  # Only on first connection
            config_bytes = await client.read_gatt_char(BLE_UUID_CONFIG)
            # _validate_crc8(config_bytes) # Uncomment to enable CRC validation

            config_str = config_bytes[:-1].decode("utf-8")
            columns = []
            for m in re.finditer(r"(\w+)\{([^}]*)\}", config_str):
                module_name = m.group(1)
                for col_def in m.group(2).split(";"):
                    if "|" in col_def:
                        col_name, type_char = col_def.split("|")
                        col_name = col_name.strip()
                        # imu/mag: use channel name as-is (e.g. acc_x, gyro_z, mag_x)
                        # other modules: prefix with module name (e.g. adc_0, adc_1)
                        if module_name not in ("imu", "mag"):
                            col_name = f"{module_name}_{col_name}"
                        columns.append((col_name, type_char.strip()))

            self.config["data"] = [c[0] for c in columns]
            # Same helper as NanoPhyling, but with the actual type from device config
            self._col_specs = [
                _make_col_spec(col_name, type_char) for col_name, type_char in columns
            ]
            self._oneDataSize = sum(s["size"] for s in self._col_specs)

            # Read rate and bufferSize from INFOS characteristic
            # Format: bytes[0]=bufferSize, bytes[1]=flags, bytes[2:4]=rate (big-endian uint16)
            infos = await client.read_gatt_char(BLE_UUID_INFOS)
            if len(infos) >= 4:
                self.config["bufferSize"] = infos[0]
                self.config["rate"] = (infos[3] << 8) | infos[2]
                print(
                    f"[{self.get_name()}] Config read: {self.config['data']} "
                    f"@ {self.config['rate']}Hz, bufferSize={self.config['bufferSize']}"
                )
            else:
                print(f"[{self.get_name()}] Config read: {self.config['data']}")

        self._init_df_if_needed()
