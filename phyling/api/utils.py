import logging
import os
import platform
import time

import urllib3


http = urllib3.PoolManager()


def get_error_message(res):
    if not res:
        return "Unexpected error"
    if type(res) is str:
        msg = res
    else:
        msg = res.data.decode("utf-8").strip()
    if msg.startswith('"') and msg.endswith('"'):
        msg = msg[1:-1]
    if len(msg) > 100:
        msg = msg[:100] + "..."
    return msg


def request(method, url, headers=None, silent=False, timeout=6, **kwargs):
    """Make a http request on API

    Args:
        method (str): GET, POST, etc.
        url (str): The url of the request
        dev_id (int): The device ID (to send header)
        headers (dict, optional): The headers. Defaults to None.
        silent (bool, optional): If True, dont log any informations. Default to False

    Returns:
        Any: The request result (or None if server not found)
    """
    if not headers:
        headers = {}
    try:
        http = urllib3.PoolManager()
        start_time = time.time()
        res = http.request(
            method=method,
            url=url,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
        if not silent:
            msg = f"[API_CALL] {time.time() - start_time:.2f}s {method} {url} - {res.status}"
            if not (res.status >= 200 and res.status <= 299):
                if res.status == 502:  # Bad gateway
                    logging.warning(msg + " (Bad gateway: Server is closed ?)")
                elif res.status == 401:
                    logging.warning(msg + " (Invalid token, need to be refreshed)")
                else:
                    msg += f" ({get_error_message(res)})"
                    logging.error(msg)
            else:
                logging.info(msg)
    except urllib3.exceptions.MaxRetryError:
        if not silent:
            logging.error(f"Unable to connect to server (not found): {method} {url}")
        if platform.system().lower() == "darwin" and url.startswith("https"):
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
