import json
import logging
import time

from phyling import phyling_utils
from phyling.api import PhylingAPI
from phyling.api.phylingSocket import PhylingSocket


class PhylingRealtime:
    """
    PhylingRealtime is a class that provides an interface for interacting with the Phyling API in realtime.
    """

    api: PhylingAPI = None
    sio: PhylingSocket = None  # type: ignore

    allDevices: list[dict] = []
    devicesStatus: dict[int, dict] = {}
    devicesIndicators: dict[int, dict] = {}
    devicesData: dict[int, dict] = {}

    def __init__(self, api: PhylingAPI):
        """
        Initializes the PhylingRealtime with the provided API key.
        :param api: An instance of the PhylingAPI class.
        """
        self.api = api
        if not self.api.is_connected():
            raise ConnectionError("Unable to connect to the Phyling API.")
        self.sio = PhylingSocket(api=self.api)

    def __str__(self) -> str:
        """
        Returns a string representation of the PhylingRealtime instance.
        :return: A string representation of the PhylingRealtime instance.
        """
        return f"PhylingRealtime({str(self.api)})"

    def selectClient(self, clientId: int) -> bool:
        """
        Selects a client by its ID.
        Available only for admin users.
        :param clientId: The ID of the client to select.
        :return: True if the client was selected successfully, False otherwise.
        """
        if "Admin" not in self.api.connected_user["roles"]:
            logging.error("Only admin users can select a client.")
            return False
        if clientId == self.api.client_id:
            return True
        self.api.client_id = int(clientId)
        return True

    """ --------------- Device list --------------- """

    def autoUpdateDeviceList(self, enabled=True) -> None:
        """
        Enables realtime updates for the selected client.
        """
        if enabled:
            self.sio.topicSubscribe(
                topic=f"app/client/{self.api.client_id}/device/list_connected",
                event="app/client/device/list_connected",
                callback=self._callbackClientDeviceConnectedList,
            )
            res = self.api.POST(
                url=f"/devices/rt/{self.api.client_id}/all",
                body="{}",
            )
            self.allDevices = json.loads(res.data.decode("utf-8"))
        else:
            self.sio.topicUnsubscribe(
                topic=f"app/client/{self.api.client_id}/device/list_connected",
                event="app/client/device/list_connected",
            )

    def getDeviceList(self) -> list[dict]:
        """
        Returns the list of devices for the selected client.
        :return: The list of devices for the selected client.
        """
        return self.allDevices

    def _callbackClientDeviceConnectedList(self, event: str, data: str) -> None:
        """
        Callback function for the "client/device/list_connected" event.
        :param event: The event name.
        :param data: The data received from the event.
        """
        self.allDevices = data

    """ --------------- Device status --------------- """

    def autoUpdateDeviceStatus(self, number: int, enabled=True) -> None:
        """
        Enables or disables realtime updates for the status of a specific device.
        :param number: The ID of the device.
        :param enabled: Whether to enable or disable the updates.
        """
        if enabled:
            self.sio.topicSubscribe(
                topic=f"app/device/{number}/board/status",
                event="app/device/board/status",
                callback=self._callbackClientDeviceStatus,
            )
            self._updateDeviceSettings(number)
        else:
            self.sio.topicUnsubscribe(
                topic=f"app/device/{number}/board/status",
                event="app/device/board/status",
            )

    def getDeviceStatus(self, number: int) -> dict | None:
        """
        Returns the status of a specific device.
        :param number: The ID of the device.
        :return: The status of the device or None if not found.
        """
        return self.devicesStatus.get(
            number,
            {
                "number": number,
                "is_connected": False,
                "state": "off",
            },
        )

    def _callbackClientDeviceStatus(self, event: str, data: str) -> None:
        """
        Callback function for the "app/device/board/status" event.
        :param event: The event name.
        :param data: The data received from the event.
        """
        number = data.get("number", None)
        if number is not None:
            self._updateDeviceStatus(number, data)

    def _updateDeviceSettings(self, number: int) -> None:
        res = self.api.GET(url=f"/devices/rt/{self.api.client_id}/{number}/settings")
        if res.status == 200:
            settings = json.loads(res.data.decode("utf-8"))
            logging.info(f"Initial settings for device {number}: {settings}")
            self.devicesStatus[number] = settings

    def _updateDeviceStatus(self, number: int, status: dict) -> None:
        """
        Updates the status of a specific device.
        :param number: The ID of the device.
        :param status: The new status of the device.
        """
        if not self.getDeviceStatus(number)["is_connected"]:
            self._updateDeviceSettings(number)
        self.devicesStatus[number] = phyling_utils.deep_merge(
            self.devicesStatus.get(number, {}), status
        )

    """ --------------- Device RPC --------------- """

    def executeRPC(
        self,
        number: int,
        method: str,
        params: dict | None = None,
        timeout: float = -1,
    ) -> dict:
        """Execute a command on a device (asynchronous, returns immediately)
        You need to send at least the method or the feature/cmd_type
        for example
        - method = v1.record.rec.start

        The command is dispatched asynchronously: this call returns immediately. When
        the command is being processed (HTTP 201) the response contains its `id`; poll
        getRPCResponse() with that `id` to retrieve the result, or use executeRPCWait().

        Args:
            number (int): The device number

        Args (json):
            method (str, optional): The command to execute on the device (start_record, restart, get_config, etc.)

            params (dict): The arguments for the command
            timeout (int, optional): The timeout for the command execution in seconds.

        Returns:
            200: The command has an immediate result (notification or cached value)
            201: The command is being processed (poll getRPCResponse() with the returned `id`)
            400: Bad request (invalid command, missing arguments, etc.)
            401: Unauthorized (user not connected)
            403: Forbidden (user does not have rights to access the device)
            404: Device not found
            418: Device not connected
            500: Internal server error (command execution failed)
            503: Service unavailable (device is turning off, etc.)
        """
        return self.api.POST(
            url=f"/devices/rt/{self.api.client_id}/{number}/rpc/request",
            body=json.dumps(
                {
                    "method": method,
                    "params": params if params else {},
                    "timeout": timeout,
                }
            ),
        )

    def getRPCResponse(self, number: int, rpc_id: str) -> dict:
        """Get the result of a command previously sent via executeRPC().

        Args:
            number (int): The device number
            rpc_id (str): The RPC request id returned by executeRPC().

        Returns:
            200: The result of the command execution
            202: The command is still being processed
            400: Bad request (missing id)
            404: Device not found, or unknown/expired request id
            500: Internal server error (command execution failed)
            503: Service unavailable (device is turning off, etc.)
        """
        return self.api.POST(
            url=f"/devices/rt/{self.api.client_id}/{number}/rpc/response",
            body=json.dumps({"id": rpc_id}),
        )

    def executeRPCWait(
        self,
        number: int,
        method: str,
        params: dict | None = None,
        timeout: float = -1,
        poll_interval: float = 1.0,
        max_wait: float = 60.0,
    ) -> dict:
        """Execute a command and wait for its result by polling /rpc/response.

        Sends the command with executeRPC(), then polls getRPCResponse() every
        `poll_interval` seconds until the result is available or `max_wait` elapses.

        Args:
            number (int): The device number
            method (str): The command to execute on the device.
            params (dict, optional): The arguments for the command.
            timeout (int, optional): The device-side timeout for the command in seconds.
            poll_interval (float, optional): Seconds between /rpc/response polls.
            max_wait (float, optional): Maximum seconds to wait for the result.

        Returns:
            The final HTTP response of getRPCResponse() (200 with the result), or the
            immediate executeRPC() response if it already carried the result, or the
            last 202 response if `max_wait` elapsed before completion.
        """
        res = self.executeRPC(number, method, params=params, timeout=timeout)
        if res.status != 201:
            return res
        rpc_id = json.loads(res.data.decode("utf-8")).get("id")
        if not rpc_id:
            return res
        waited = 0.0
        while waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval
            res = self.getRPCResponse(number, rpc_id)
            if res.status != 202:
                return res
        return res

    """ --------------- Device realtime indicator --------------- """

    def autoUpdateDeviceIndicator(self, number: int, enabled=True) -> None:
        """
        Enables or disables realtime updates for the indicator of a specific device.
        :param number: The ID of the device.
        :param enabled: Whether to enable or disable the updates.
        """
        if enabled:
            self.sio.topicSubscribe(
                topic=f"app/device/{number}/ind/json/all",
                event="app/device/ind/json/all",
                callback=self._callbackClientDeviceIndicator,
            )
        else:
            self.sio.topicUnsubscribe(
                topic=f"app/device/{number}/ind/json/all",
                event="app/device/ind/json/all",
            )

    def getDeviceIndicator(self, number: int) -> dict | None:
        """
        Returns the indicator of a specific device.
        :param number: The ID of the device.
        :return: The indicator of the device or {} if not found.
        """
        return self.devicesIndicators.get(
            number,
            {
                "number": number,
                "recTime": -1,
                "indicators": {},
            },
        )

    def _callbackClientDeviceIndicator(self, event: str, data: str) -> None:
        """
        Callback function for the "device/ind/realtime" event.
        :param event: The event name.
        :param data: The data received from the event.
        """
        number = data.get("number", None)
        if number is not None:
            self.devicesIndicators[number] = data

    """ --------------- Device realtime data --------------- """

    def autoUpdateDeviceData(self, number: int, enabled=True) -> None:
        """
        Enables or disables realtime updates for the data of a specific device.
        :param number: The ID of the device.
        :param enabled: Whether to enable or disable the updates.
        """
        if enabled:
            self.sio.topicSubscribe(
                topic=f"app/device/{number}/data/json/all",
                event="app/device/data/json/all",
                callback=self._callbackClientDeviceData,
            )
        else:
            self.sio.topicUnsubscribe(
                topic=f"app/device/{number}/data/json/all",
                event="app/device/data/json/all",
            )

    def getDeviceData(self, number: int) -> dict | None:
        """
        Returns the data of a specific device.
        :param number: The ID of the device.
        :return: The data of the device or {} if not found.
        """
        return self.devicesData.get(
            number,
            {
                "number": number,
                "recTime": -1,
                "data": {},
                "selections": [],
            },
        )

    def _callbackClientDeviceData(self, event: str, data: str) -> None:
        """
        Callback function for the "app/device/data/json/all" event.
        :param event: The event name.
        :param data: The data received from the event.
        """
        number = data.get("number", None)
        if number is not None:
            self.devicesData[number] = data
