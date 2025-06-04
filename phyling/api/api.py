import logging
import os
import platform
import time
from typing import Union

import ujson
import urllib3
from urllib3.filepost import encode_multipart_formdata

from phyling.api import utils
from phyling.api.record import Record
from phyling.api.user import User


class PhylingAPI:
    """
    PhylingAPI is a class that provides an interface for interacting with the Phyling API.
    """

    mail = None
    password = None
    baseurl = None

    _http = None
    _access_token = None
    _refresh_token = None
    _is_connected = False

    def __init__(
        self, mail: str, password: str, url: str = "https://api.app.phyling.fr"
    ):
        """
        Initializes the PhylingAPI with the provided API key.
        :param mail: The email address of the user.
        :param password: The password of the user.
        :param baseurl: The URL of the Phyling API. Default is "app.phyling.fr".
        """
        self.mail = mail
        self.password = password
        self.baseurl = url
        if self.baseurl.endswith("/"):
            self.baseurl = self.baseurl[:-1]
        self._http = urllib3.PoolManager()
        self.login()

    def __str__(self) -> str:
        """
        Returns a string representation of the PhylingAPI instance.
        :return: A string representation of the PhylingAPI instance.
        """
        return (
            f"PhylingAPI(mail={self.mail}, baseurl={self.baseurl})"
            f" -> {'Connected' if self._is_connected else 'Not connected'}"
        )

    def is_connected(self) -> bool:
        """
        Checks if the PhylingAPI is connected.
        :return: True if connected, False otherwise.
        """
        return self._is_connected

    def login(self) -> bool:
        """
        Logs in to the Phyling API using the provided email and password.
        :return: True if login is successful, False otherwise.
        """
        res = self.request(
            auto_login=False,
            method="POST",
            url="/login",
            headers={
                "Content-Type": "application/json",
            },
            body=ujson.dumps(
                {
                    "mail": self.mail,
                    "password": self.password,
                }
            ),
            timeout=3,
            retries=False,
        )
        if not res:
            self._is_connected = False
            return False
        if res.status != 200:
            self._is_connected = False
            return False

        response_data = ujson.loads(res.data)
        self._access_token = response_data.get("access_token")
        self._refresh_token = response_data.get("refresh_token")
        if not self._is_connected:
            self._is_connected = True
        return True

    def refresh_token(self) -> bool:
        """
        Refreshes the access token using the refresh token.
        :return: True if token refresh is successful, False otherwise.
        """
        res = self.request(
            auto_login=False,
            method="POST",
            url="/refresh/mallette",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._refresh_token}",
            },
            timeout=3,
            retries=False,
        )
        if not res:
            self._is_connected = False
            return False

        if res.status == 200:
            response_data = ujson.loads(res.data)
            self._access_token = response_data.get("access_token")
        else:
            if not self.login():
                self._is_connected = False
                return False

        self._is_connected = True
        return True

    def request(
        self,
        method,
        url,
        headers=None,
        input_path=None,
        silent=False,
        timeout=12,
        auto_login=True,
        **kwargs,
    ) -> Union[urllib3.HTTPResponse, None]:
        """Make a http request on API

        Args:
            method (str): GET, POST, etc.
            url (str): The url of the request
            dev_id (int): The device ID (to send header)
            headers (dict, optional): The headers. Defaults to None.
            input_path (str, optional): The path to the input file. Defaults to None.
            silent (bool, optional): If True, dont log any informations. Default to False
            timeout (int, optional): The timeout for the request. Default to 12 seconds.
            auto_login (bool, optional): If False, dont automatically login. Used for login request
            **kwargs: Additional arguments for the request.

        Returns:
            Any: The request result (or None if server not found)
        """
        if not url.startswith("http"):
            url = self.baseurl + url
        elif not url.startswith("/"):
            url = "/" + url
        if not self._access_token and auto_login:
            if not self.login():
                return None
        if not headers:
            headers = {}
        if "Authorization" not in headers and auto_login:
            headers["Authorization"] = f"Bearer {self._access_token}"

        if input_path:
            extension = input_path.split(".")[-1]
            with open(input_path, "rb") as f:
                fields = {
                    "file": (f"file.{extension}", f.read()),
                    "data": kwargs["body"],
                }
                encoded_data, content_type = encode_multipart_formdata(fields)
                headers["Content-Type"] = content_type
                kwargs["body"] = encoded_data

        if "body" in kwargs and type(kwargs["body"]) is dict:
            kwargs["body"] = ujson.dumps(kwargs["body"])

        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        try:
            start_time = time.time()
            res = self._http.request(
                method=method,
                url=url,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
            if res.status == 401 and auto_login:
                if not self.refresh_token():
                    return None
                headers["Authorization"] = f"Bearer {self._access_token}"
                res = self._http.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )

            if not silent:
                msg = f"[API_CALL] {time.time() - start_time:.2f}s - {method} {url} - {res.status}"
                if not (res.status >= 200 and res.status <= 299):
                    if res.status == 502:  # Bad gateway
                        logging.warning(msg + " (Bad gateway: Server is closed ?)")
                    elif res.status == 401 and auto_login:
                        logging.warning(msg + " (Invalid token, need to be refreshed)")
                    else:
                        msg += f" ({utils.get_error_message(res)})"
                        logging.error(msg)
                else:
                    logging.info(msg)
        except urllib3.exceptions.MaxRetryError:
            if not silent:
                logging.error(
                    f"Unable to connect to server (not found): {method} {url}"
                )
            if platform.system().lower() == "darwin":
                logging.info("Reinstall SSL certificate")
                os.system("/Applications/Python\\ 3.10/Install\\ Certificates.command")
            return None
        except urllib3.exceptions.ReadTimeoutError:
            if not silent:
                logging.error(f"Unable to connect to server (timeout): {method} {url}")
            return None
        except Exception as e:
            if not silent:
                logging.error(f"Unable to connect to server ({str(e)}): {method} {url}")
            return None
        return res

    def GET(
        self,
        url,
        headers=None,
        input_path=None,
        silent=False,
        timeout=12,
        auto_login=True,
        **kwargs,
    ) -> Union[urllib3.HTTPResponse, None]:
        """Make a GET request on API

        Args:
            url (str): The url of the request
            headers (dict, optional): The headers. Defaults to None.
            input_path (str, optional): The path to the input file. Defaults to None.
            silent (bool, optional): If True, dont log any informations. Default to False
            timeout (int, optional): The timeout for the request. Default to 12 seconds.
            auto_login (bool, optional): If False, dont automatically login. Used for login request
            **kwargs: Additional arguments for the request.

        Returns:
            Any: The request result (or None if server not found)
        """
        return self.request(
            method="GET",
            url=url,
            headers=headers,
            input_path=input_path,
            silent=silent,
            timeout=timeout,
            auto_login=auto_login,
            **kwargs,
        )

    def POST(
        self,
        url,
        headers=None,
        input_path=None,
        silent=False,
        timeout=12,
        auto_login=True,
        **kwargs,
    ) -> Union[urllib3.HTTPResponse, None]:
        """Make a POST request on API

        Args:
            url (str): The url of the request
            headers (dict, optional): The headers. Defaults to None.
            input_path (str, optional): The path to the input file. Defaults to None.
            silent (bool, optional): If True, dont log any informations. Default to False
            timeout (int, optional): The timeout for the request. Default to 12 seconds.
            auto_login (bool, optional): If False, dont automatically login. Used for login request
            **kwargs: Additional arguments for the request.

        Returns:
            Any: The request result (or None if server not found)
        """
        return self.request(
            method="POST",
            url=url,
            headers=headers,
            input_path=input_path,
            silent=silent,
            timeout=timeout,
            auto_login=auto_login,
            **kwargs,
        )

    def PUT(
        self,
        url,
        headers=None,
        input_path=None,
        silent=False,
        timeout=12,
        auto_login=True,
        **kwargs,
    ) -> Union[urllib3.HTTPResponse, None]:
        """Make a PUT request on API

        Args:
            url (str): The url of the request
            headers (dict, optional): The headers. Defaults to None.
            input_path (str, optional): The path to the input file. Defaults to None.
            silent (bool, optional): If True, dont log any informations. Default to False
            timeout (int, optional): The timeout for the request. Default to 12 seconds.
            auto_login (bool, optional): If False, dont automatically login. Used for login request
            **kwargs: Additional arguments for the request.

        Returns:
            Any: The request result (or None if server not found)
        """
        return self.request(
            method="PUT",
            url=url,
            headers=headers,
            input_path=input_path,
            silent=silent,
            timeout=timeout,
            auto_login=auto_login,
            **kwargs,
        )

    def DELETE(
        self,
        url,
        headers=None,
        input_path=None,
        silent=False,
        timeout=12,
        auto_login=True,
        **kwargs,
    ) -> Union[urllib3.HTTPResponse, None]:
        """Make a DELETE request on API

        Args:
            url (str): The url of the request
            headers (dict, optional): The headers. Defaults to None.
            input_path (str, optional): The path to the input file. Defaults to None.
            silent (bool, optional): If True, dont log any informations. Default to False
            timeout (int, optional): The timeout for the request. Default to 12 seconds.
            auto_login (bool, optional): If False, dont automatically login. Used for login request
            **kwargs: Additional arguments for the request.

        Returns:
            Any: The request result (or None if server not found)
        """
        return self.request(
            method="DELETE",
            url=url,
            headers=headers,
            input_path=input_path,
            silent=silent,
            timeout=timeout,
            auto_login=auto_login,
            **kwargs,
        )

    def get_users(
        self,
        client_id: Union[int, None] = None,
        group_ids: list = [],
        role: Union[str, None] = None,
        active: bool = True,
        search: str = "",
        pageId: int = 1,
        pageSize: int = 0,
        soft: bool = False,
    ) -> Union[dict, None]:
        """Get the users from the API

        Args:
            client_id (int): clients id. None to select all
            group_ids (list): list of group ids. Empty to select all
            role (str, optional): The users roles (Admin, Coach or Athlete). None to select all
            active (bool, optional): The users active status. Defaults to True.
            search (str): The search string
            pageSize (int): Size of one page. Default: -1 (select all)
            pageId (int): Page id. Default: 1
            soft (bool): If true, only return minimal information on users

        Returns:
            dict: {
                "items": [User, ...],
                "total": int,
            }
        """
        res = self.POST(
            url="/users/all",
            body=ujson.dumps(
                {
                    "client_id": client_id,
                    "group_ids": group_ids,
                    "role": role,
                    "active": active,
                    "search": search,
                    "pageId": pageId,
                    "pageSize": pageSize,
                    "soft": soft,
                }
            ),
        )
        if not res:
            return None
        if res.status != 200:
            return None
        response_data = ujson.loads(res.data)
        res = {
            "items": [
                User(api=self, desc=rec) for rec in response_data.get("items", [])
            ],
            "total": response_data.get("total", 0),
        }
        return res

    def get_records(
        self,
        type: str = "all",
        pageSize: int = 10,
        pageId: int = 1,
        onlyFavorite: bool = False,
        userIds: list = [],
        deviceIds: list = [],
        clientIds: list = [],
        groupIds: list = [],
        exerciseIds: list = [],
        sportIds: list = [],
        record_type: str = "",
        scenarioIds: list = [],
        minDate: str = "",
        maxDate: str = "",
    ) -> Union[dict, None]:
        """Get the records from the API

        Args:
            type (str): should be `all`, `new` or `associated`
            pageSize (int): Size of one page. Default: -1 (select all)
            pageId (int): Page id. Default: 1
            onlyFavorite (bool, optional): Select only favorites. Default to False.
            userIds (list): Users ids. Empty to select all
            deviceIds (list): devices ids. Empty to select all
            clientIds (list): clients ids. Empty to select all
            groupIds (list): groups ids. Empty to select all
            exerciseIds (list): exercises ids. Empty to select all
            sportIds (list): sport ids. Empty to select all
            record_type (str): record type. Can be seance, record, scenario, calib, miniphyling, fusion or video
            scenarioIds (list[int]): scenario ids. Empty to select all
            minDate (str): min record date. Empty to select all
            maxDate (str): max record date. Empty to select all

        Returns:
            dict: {
                "records": [Record, ...],
                "total": int,
            }
        """
        res = self.POST(
            url="/records/all",
            body=ujson.dumps(
                {
                    "type": type,
                    "pageSize": pageSize,
                    "pageId": pageId,
                    "onlyFavorite": onlyFavorite,
                    "userIds": userIds,
                    "deviceIds": deviceIds,
                    "clientIds": clientIds,
                    "groupIds": groupIds,
                    "exerciseIds": exerciseIds,
                    "sportIds": sportIds,
                    "record_type": record_type,
                    "scenario_ids": scenarioIds,
                    "minDate": minDate,
                    "maxDate": maxDate,
                }
            ),
        )
        if not res:
            return None
        if res.status != 200:
            return None
        response_data = ujson.loads(res.data)
        res = {
            "records": [
                Record(api=self, desc=rec) for rec in response_data.get("records", [])
            ],
            "total": response_data.get("total", 0),
        }
        return res

    def download_record(
        self,
        rec_id: int,
        file_type: str,
        download_path: str,
        overwrite: bool = True,
        timeout: int = 180,
        **kwargs,
    ) -> bool:
        """
        Download the record.

        Args:
            rec_id (int): The ID of the record to download.
            file_type (str): The type of file to download. Can be one of the following:
                - "raw": raw data (.txt)
                - "decoded": decoded data (.csv)
                - "pdf": pdf report (.pdf)
                - "stats": stats report (.csv)
                - "specific_stats": specific stats report (.csv)
                - "compressed": compressed data (.csv)
                - "video": video file (.mp4) -> only for video records
                - "zip": zip file (.zip)
            download_path (str): The path to save the downloaded file.
            overwrite (bool): If True, overwrite the file if it already exists. Default is False.
            timeout (int): The timeout for the request. Default is 180 seconds.
            **kwargs: Additional arguments for the request.
        """
        if os.path.exists(download_path) and not overwrite:
            logging.error("File already exists and overwrite is set to False")
            return False

        res = self.request(
            method="POST",
            url=f"/records/{rec_id}/file/{file_type}",
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            body=ujson.dumps(kwargs),
        )
        if not res:
            return False
        else:
            if file_type == "pdf" and res.status == 201:
                task_id = ujson.loads(res.data.decode("utf-8"))["task_id"]
                return self.wait_for_task(
                    task_id=task_id,
                    overwrite=overwrite,
                    download_path=download_path,
                    timeout=timeout,
                    check_interval=3,
                )

            if res.status == 200:
                if os.path.exists(download_path):
                    os.remove(download_path)
                with open(download_path, "wb") as f:
                    f.write(res.data)
                logging.info(f"File downloaded to: {download_path}")
                return True
        return False

    def add_task(
        self,
        name: str,
        wait_for_finish: bool = True,
        task_args: dict = {},
        download_path: Union[str, None] = None,
        overwrite: bool = False,
        timeout: int = 180,
        check_interval: int = 3,
        **kwargs,
    ) -> bool:
        """
        Add a task to the app (decode, create backup, etc.)

        Args:
            name (str): The task name.
            wait_for_finish (bool): If True, wait for the task to finish. Default is True.
            task_args (dict): The task arguments. example: {"rec_id": 1234, "args": {"overwrite": True}}
            download_path (str): The path to save the downloaded file.
            overwrite (bool): If True, overwrite the file if it already exists. Default is False.
            timeout (int): The timeout for the request. Default is 180 seconds.
            check_interval (int): The interval between checks for the task status. Default is 3 seconds.
            **kwargs: Additional arguments for the request.
        """
        if (
            download_path is not None
            and os.path.exists(download_path)
            and not overwrite
        ):
            logging.error("File already exists and overwrite is set to False")
            return False

        res = self.request(
            method="POST",
            url="/tasks",
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            body=ujson.dumps(
                {
                    "name": name,
                    **task_args,
                }
            ),
        )
        if not res:
            return False
        if wait_for_finish:
            task_id = ujson.loads(res.data.decode("utf-8"))["task_id"]
            return self.wait_for_task(
                task_id=task_id,
                overwrite=overwrite,
                download_path=download_path,
                timeout=timeout,
                check_interval=check_interval,
            )
        else:
            if res.status == 200:
                return True
        return False

    def add_task_with_file(
        self,
        name: str,
        filename: str,
        task_args: dict = {},
        wait_for_finish: bool = True,
        overwrite: bool = False,
        timeout: int = 180,
        check_interval: int = 3,
        **kwargs,
    ) -> bool:
        """
        Add a task to the app with a file in the request (upload backup, etc.)

        Args:
            name (str): The task name.
            wait_for_finish (bool): If True, wait for the task to finish. Default is True.
            task_args (dict): The task arguments. example: {"rec_id": 1234, "args": {"overwrite": True}}
            download_path (str): The path to save the downloaded file.
            overwrite (bool): If True, overwrite the file if it already exists. Default is False.
            timeout (int): The timeout for the request. Default is 180 seconds.
            check_interval (int): The interval between checks for the task status. Default is 3 seconds.
            **kwargs: Additional arguments for the request.
        """
        if not os.path.exists(filename):
            logging.error("File already exists and overwrite is set to False")
            return False

        res = self.request(
            method="POST",
            url="/tasks/with_file",
            input_path=filename,
            timeout=timeout,
            body=ujson.dumps(
                {
                    "name": name,
                    **task_args,
                }
            ),
        )
        if not res:
            return False
        if wait_for_finish:
            task_id = ujson.loads(res.data.decode("utf-8"))["task_id"]
            return self.wait_for_task(
                task_id=task_id,
                overwrite=overwrite,
                timeout=timeout,
                check_interval=check_interval,
            )
        else:
            if res.status == 200:
                return True
        return False

    def wait_for_task(
        self,
        task_id: int,
        download_path: Union[str, None] = None,
        overwrite: bool = False,
        timeout: int = 300,
        check_interval: int = 3,
    ) -> bool:
        time_end = time.time() + timeout
        while time.time() < time_end:
            task_status = self.request(
                method="GET",
                url=f"/tasks/status?id={task_id}",
                headers={"Content-Type": "application/json"},
                timeout=2,
            )
            task_info = ujson.loads(task_status.data.decode("utf-8"))
            if len(task_info) == 0:
                return False
            task_info = task_info[0]
            if task_info["error"]:
                return False
            elif task_info["success"]:
                if download_path is not None and "download" in task_info["result"]:
                    res = self.request(
                        method="POST",
                        url=f"/tasks/{task_id}/download",
                        headers={"Content-Type": "application/json"},
                        timeout=2,
                    )
                    if res is None or res.status not in (200, 204):
                        return False

                    if os.path.exists(download_path):
                        if overwrite:
                            os.remove(download_path)
                        else:
                            logging.error(
                                f"File already exists and overwrite is set to False: {download_path}"
                            )
                            return False

                    if res.status == 204:
                        download_url = res.headers.get("X-Download-Url")
                        if download_url is None:
                            return False
                        res = utils.request(
                            method="GET",
                            url=download_url,
                            timeout=2,
                        )
                        if res is None or res.status != 200:
                            logging.error(
                                f"Cannot download result from S3: {download_url}"
                            )
                            return False
                        # save the file
                        with open(download_path, "wb") as f:
                            f.write(res.data)
                    else:
                        with open(download_path, "wb") as f:
                            f.write(res.data)
                return True
            time.sleep(check_interval)
        logging.error(f"Task {task_id} timed out after {timeout} seconds")
        return False
