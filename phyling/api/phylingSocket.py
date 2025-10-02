import logging
from typing import Any
from typing import Dict
from typing import Optional

import socketio

from phyling.api.api import PhylingAPI


sio = socketio.Client()
sio_callbacks = {}  # {"event": callback, ...}
"""
sio_callbacks = {
    "connect": on_connect,
    "disconnect": on_disconnect,
    "client/device/all": on_client_device_all,
    "device/data/realtime": on_device_data_realtime,
}
"""


@sio.on("connect")
def on_connect():
    if "connect" in sio_callbacks:
        sio_callbacks["connect"]()


@sio.on("disconnect")
def on_disconnect():
    if "disconnect" in sio_callbacks:
        sio_callbacks["disconnect"]()


@sio.on("*")
def catch_all(event, data):
    if event in sio_callbacks:
        sio_callbacks[event](event=event, data=data)
    else:
        print(f"Catch-all event '{event}' received with data: {data}")


class PhylingSocket:
    """Manage socket.io connectivity and authentication workflows."""

    api: PhylingAPI = None
    socket_rooms: Dict[str, int] = {}

    def __init__(
        self,
        api: PhylingAPI,
    ) -> None:
        self.api = api
        self.socket_rooms = {}
        sio_callbacks["connect"] = self._on_connect
        sio_callbacks["disconnect"] = self._on_disconnect
        sio_callbacks["message/info"] = self._on_message_info
        sio_callbacks["message/error"] = self._on_message_error
        sio.connect(self.api.baseurl, transports=["websocket"])

    def _on_connect(self) -> None:
        logging.info("Socket connected to %s", self.api.baseurl)
        for room, counter in self.socket_rooms.items():
            if counter > 0:
                sio.emit(
                    "subscribe",
                    {"room": room, "token": self.api._access_token},
                )

    def _on_disconnect(self) -> None:
        logging.warning("Socket disconnected from %s", self.api.baseurl)

    def _on_message_info(self, event: str, data: str) -> None:
        logging.info(f"Socket message info: {data}")

    def _on_message_error(self, event: str, data: str) -> None:
        logging.error(f"Socket message error: {data}")

    def emit(self, event: str, data: Any, *, namespace: Optional[str] = None) -> None:
        sio.emit(event, data, namespace=namespace)

    def topicSubscribe(self, topic: str, event: str, callback: callable) -> None:
        if topic not in self.socket_rooms or self.socket_rooms[topic] <= 0:
            self.socket_rooms[topic] = 1
            if sio.connected:
                sio.emit(
                    "subscribe",
                    {"room": topic, "token": self.api._access_token},
                )
        else:
            self.socket_rooms[topic] += 1

        sio_callbacks[event] = callback

    def topicUnsubscribe(self, topic: str, event: str) -> None:
        if topic not in self.socket_rooms:
            return
        if event in sio_callbacks:
            del sio_callbacks[event]
        self.socket_rooms[topic] -= 1
        if self.socket_rooms[topic] <= 0:
            if sio.connected:
                sio.emit(
                    "unsubscribe",
                    {"room": topic, "token": self.api._access_token},
                )
