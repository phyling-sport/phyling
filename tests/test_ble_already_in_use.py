import asyncio
import contextlib
import io
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bleak import BleakClient

from phyling.ble.base_device import _make_col_spec
from phyling.ble.base_device import ALREADY_IN_USE_WINDOW_SEC
from phyling.ble.base_device import BaseDevice
from phyling.ble.base_device import BLE_UUID_PHYLING
from phyling.ble.base_device import is_already_in_use_disconnection

REFUSAL = "[NanoPhyling_42] already in use by another device (Maxi or app), connection refused."
# One notification: 8-byte timestamp (us), 2-byte spacing (us), one int16 sample
NOTIFICATION = (1_000_000).to_bytes(8, "little") + (5_000).to_bytes(2, "little") + (100).to_bytes(2, "little")


class TestIsAlreadyInUseDisconnection(unittest.TestCase):
    def test_refusal_within_window(self):
        self.assertTrue(is_already_in_use_disconnection(0.12, notified=False, client_requested=False))

    def test_refusal_at_window_edge(self):
        self.assertTrue(is_already_in_use_disconnection(ALREADY_IN_USE_WINDOW_SEC, False, False))

    def test_link_loss_after_notification(self):
        self.assertFalse(is_already_in_use_disconnection(0.5, notified=True, client_requested=False))

    def test_link_loss_after_window(self):
        self.assertFalse(is_already_in_use_disconnection(ALREADY_IN_USE_WINDOW_SEC + 0.01, False, False))

    def test_disconnection_before_subscription(self):
        self.assertFalse(is_already_in_use_disconnection(None, notified=False, client_requested=False))

    def test_client_requested(self):
        self.assertFalse(is_already_in_use_disconnection(0.12, notified=False, client_requested=True))


class FakeDevice(BaseDevice):
    """Minimal device with one int16 column, no GATT access during setup."""

    async def _setup_config(self, client: BleakClient) -> None:
        """Build a single-column config."""
        self.config["data"] = ["acc_x"]
        self._col_specs = [_make_col_spec("acc_x", "B")]
        self._oneDataSize = 2
        self._init_df_if_needed()


class FakeBleakClient:
    """BleakClient stand-in: runs a scripted notification and remote disconnection after subscription."""

    notify_after: float | None = None
    remote_disconnect_after: float | None = None
    connections = 0

    def __init__(self, address, disconnected_callback=None):
        self._disconnected_callback = disconnected_callback
        self._connected = True
        self.services = [SimpleNamespace(characteristics=[SimpleNamespace(uuid=BLE_UUID_PHYLING)])]

    async def __aenter__(self):
        type(self).connections += 1
        return self

    async def __aexit__(self, *exc):
        self._disconnect()

    def _disconnect(self):
        if self._connected:
            self._connected = False
            self._disconnected_callback(self)

    async def start_notify(self, uuid, handler):
        loop = asyncio.get_running_loop()
        if self.notify_after is not None:
            loop.call_later(self.notify_after, handler, uuid, NOTIFICATION)
        if self.remote_disconnect_after is not None:
            loop.call_later(self.remote_disconnect_after, self._disconnect)

    async def write_gatt_char(self, uuid, data, response=None):
        pass

    async def stop_notify(self, uuid):
        pass


def run_device(notify_after: float | None, remote_disconnect_after: float | None, duration: float):
    """
    Run one FakeDevice recording against a scripted FakeBleakClient.

    :return: (device, printed output, number of connections, elapsed seconds)
    """
    client_cls = type(
        "ScriptedClient",
        (FakeBleakClient,),
        {"notify_after": notify_after, "remote_disconnect_after": remote_disconnect_after, "connections": 0},
    )
    device = FakeDevice(ble_name="NanoPhyling_42", address="AA:BB:CC:DD:EE:FF")
    out = io.StringIO()
    start = time.monotonic()
    with patch("phyling.ble.base_device.BleakClient", client_cls), contextlib.redirect_stdout(out):
        asyncio.run(device._run_ble_client(duration))
    return device, out.getvalue(), client_cls.connections, time.monotonic() - start


class TestRunBleClientAlreadyInUse(unittest.TestCase):
    def test_refusal_reported_once_without_1s_reconnection(self):
        _, out, connections, elapsed = run_device(notify_after=None, remote_disconnect_after=0.12, duration=1.5)
        self.assertEqual(out.count(REFUSAL), 1)
        self.assertIn(f"{REFUSAL} Retrying in 15s...", out)
        self.assertNotIn("Reconnecting in 1s", out)
        self.assertEqual(connections, 1)
        self.assertLess(elapsed, 2.5)

    def test_refusal_retried_after_delay(self):
        with patch("phyling.ble.base_device.ALREADY_IN_USE_RETRY_SEC", 0.3):
            _, out, connections, _ = run_device(notify_after=None, remote_disconnect_after=0.12, duration=1.2)
        self.assertGreaterEqual(connections, 2)
        self.assertEqual(out.count(REFUSAL), connections)

    def test_link_loss_after_notification_not_refusal(self):
        device, out, _, _ = run_device(notify_after=0.05, remote_disconnect_after=0.12, duration=0.5)
        self.assertNotIn(REFUSAL, out)
        self.assertIn("Unexpected disconnect. Reconnecting in 1s...", out)
        self.assertEqual(device.nbDatas, 1)

    def test_client_requested_disconnection_not_refusal(self):
        device, out, _, _ = run_device(notify_after=None, remote_disconnect_after=None, duration=0.3)
        self.assertNotIn(REFUSAL, out)
        self.assertTrue(device.disconnect)


if __name__ == "__main__":
    unittest.main()
