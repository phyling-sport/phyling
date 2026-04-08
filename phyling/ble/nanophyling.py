from typing import Any
from typing import Union

import ujson
from bleak import BleakClient

from phyling.ble.base_device import _make_col_spec
from phyling.ble.base_device import BaseDevice
from phyling.ble.base_device import BLE_NOTIF_NANO_SEND_CONFIG
from phyling.ble.base_device import BLE_UUID_MAXI_VERSION
from phyling.ble.base_device import BLE_UUID_PHYLING
from phyling.ble.base_device import BLE_UUID_VERSION

NANO_DEF_CONFIG = {
    "rate": 200,
    "bufferSize": 200,
    "data": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
}


class NanoPhyling(BaseDevice):

    def __init__(
        self,
        ble_name: Union[str, None],
        address: Union[str, None] = None,
        config: dict[str, Any] = NANO_DEF_CONFIG,
        module_name: Union[str, None] = None,
    ):
        """
        BLE client for Nano-Phyling devices. Provide ble_name OR address.

        :param ble_name: BLE device name (e.g. "NanoPhyling_42")
        :param address: BLE device address
        :param config: Configuration dictionary (default: NANO_DEF_CONFIG)
        :param module_name: Custom module name (overrides auto-generated name in logs and df prefixes)
        """
        super().__init__(ble_name, address, module_name)
        self.config = {**NANO_DEF_CONFIG, **config}

    async def _setup_config(self, client: BleakClient) -> None:
        """Send version and config JSON to the device, then build _col_specs."""
        await client.write_gatt_char(BLE_UUID_MAXI_VERSION, b"v7.0.4", response=True)
        await client.write_gatt_char(
            BLE_UUID_PHYLING,
            BLE_NOTIF_NANO_SEND_CONFIG + bytes(ujson.dumps(self.config), "utf-8"),
        )
        await client.read_gatt_char(BLE_UUID_VERSION)
        # All Nano columns are int16 (type_char="B") — same helper as MiniPhyling
        self._col_specs = [_make_col_spec(col, "B") for col in self.config["data"]]
        self._oneDataSize = sum(s["size"] for s in self._col_specs)
        self._init_df_if_needed()
