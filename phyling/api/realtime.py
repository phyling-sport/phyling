import json
import logging

from phyling import phyling_utils
from phyling.api import PhylingAPI
from phyling.api.phylingSocket import PhylingSocket


class PhylingRealtime:
    """
    PhylingRealtime is a class that provides an interface for interacting with the Phyling API in realtime.
    """

    api: PhylingAPI = None
    sio: PhylingSocket = None  # type: ignore
    user: dict | None = None
    clientId: int = 0

    allDevices: list[dict] = []
    devicesStatus: dict[int, dict] = {}
    devicesIndicators: dict[int, dict] = {}
    devicesData: dict[int, dict] = {}

    def __init__(self, api: PhylingAPI):
        """
        Initializes the PhylingRealtime with the provided API key.
        :param mail: The email address of the user.
        :param password: The password of the user.
        :param baseurl: The URL of the Phyling API. Default is "app.phyling.fr".
        """
        self.api = api
        if not self.api.is_connected():
            self.login()
        if not self.api.is_connected():
            raise ConnectionError("Unable to connect to the Phyling API.")
        res = api.GET(url="/login")
        self.user = json.loads(res.data.decode("utf-8"))
        if not self.user:
            raise ConnectionError(
                "Unable to retrieve user information from the Phyling API."
            )
        self.clientId = self.user["client_id"]

        self.sio = PhylingSocket(api=self.api)

    def __str__(self) -> str:
        """
        Returns a string representation of the PhylingRealtime instance.
        :return: A string representation of the PhylingRealtime instance.
        """
        return (
            f"PhylingRealtime(mail={self.api.mail}, baseurl={self.api.baseurl})"
            f" -> {'Connected' if self.api.is_connected() else 'Not connected'}"
        )

    def selectClient(self, clientId: int) -> bool:
        """
        Selects a client by its ID.
        Available only for admin users.
        :param clientId: The ID of the client to select.
        :return: True if the client was selected successfully, False otherwise.
        """
        if "Admin" not in self.user["roles"]:
            logging.error("Only admin users can select a client.")
            return False
        if clientId == self.clientId:
            return True
        self.clientId = int(clientId)
        return True

    """ --------------- Device list --------------- """

    def autoUpdateDeviceList(self, enabled=True) -> None:
        """
        Enables realtime updates for the selected client.
        """
        if enabled:
            self.sio.topicSubscribe(
                topic=f"app/client/{self.clientId}/device/connected_list",
                event="app/client/device/connected_list",
                callback=self._callbackClientDeviceConnectedList,
            )
            res = self.api.POST(
                url=f"/devices/rt/{self.clientId}/all",
                body="{}",
            )
            self.allDevices = json.loads(res.data.decode("utf-8"))
        else:
            self.sio.topicUnsubscribe(
                topic=f"app/client/{self.clientId}/device/connected_list",
                event="app/client/device/connected_list",
            )

    def getDeviceList(self) -> list[dict]:
        """
        Returns the list of devices for the selected client.
        :return: The list of devices for the selected client.
        """
        return self.allDevices

    def _callbackClientDeviceConnectedList(self, event: str, data: str) -> None:
        """
        Callback function for the "client/device/connected_list" event.
        :param event: The event name.
        :param data: The data received from the event.
        """
        data = json.loads(data)
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
        data = json.loads(data)
        number = data.get("number", None)
        if number is not None:
            self._updateDeviceStatus(number, data)

    def _updateDeviceSettings(self, number: int) -> None:
        res = self.api.GET(url=f"/devices/rt/{self.clientId}/{number}/settings")
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
        wait_for_result: bool = True,
    ) -> dict:
        """Execute a command on a device
        You need to send at least the method or the feature/cmd_type
        for example
        - method = v1.record.rec.start

        Args:
            number (int): The device number

        Args (json):
            method (str, optional): The command to execute on the device (start_record, restart, get_config, etc.)

            params (dict): The arguments for the command
            timeout (int, optional): The timeout for the command execution in seconds.
            wait_for_result (bool, optional): If True, wait for the command to be executed and return the result.
                                            If False, return immediately.

        Returns:
            200: The result of the command execution
            400: Bad request (invalid command, missing arguments, etc.)
            401: Unauthorized (user not connected)
            403: Forbidden (user does not have rights to access the device)
            404: Device not found
            418: Device not connected
            500: Internal server error (command execution failed)
            503: Service unavailable (device is turning off, etc.)
        """
        return self.api.POST(
            url=f"/devices/rt/{self.clientId}/{number}/rpc/request",
            body=json.dumps(
                {
                    "method": method,
                    "params": params if params else {},
                    "timeout": timeout,
                    "wait_for_result": wait_for_result,
                }
            ),
        )

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
        data = json.loads(data)
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
        data = json.loads(data)
        number = data.get("number", None)
        if number is not None:
            self.devicesData[number] = data
