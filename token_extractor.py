from abc import ABC, abstractmethod
import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import random
import re
import socket
import sys
import tempfile
import threading
import time
from getpass import getpass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import requests
from colorama import Fore, Style, init

try:
    from Crypto.Cipher import ARC4
except ModuleNotFoundError:
    from Cryptodome.Cipher import ARC4

from PIL import Image

if sys.platform != "win32":
    import readline

SERVERS = ["cn", "de", "us", "ru", "tw", "sg", "in", "i2"]
NAME_TO_LEVEL = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-ni", "--non_interactive", required=False, help="Non-interactive mode", action="store_true")
    parser.add_argument("-u", "--username", required=False, help="Username")
    parser.add_argument("-p", "--password", required=False, help="Password")
    parser.add_argument("-s", "--server", required=False, help="Server", choices=[*SERVERS, ""])
    parser.add_argument("-l", "--log_level", required=False, help="Log level", default="CRITICAL",
                        choices=list(NAME_TO_LEVEL.keys()))
    parser.add_argument("-o", "--output", required=False, help="Output file")
    parser.add_argument("--host", required=False, help="Host")
    return parser


class ColorFormatter(logging.Formatter):
    COLORS = {
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "FATAL": Fore.RED + Style.BRIGHT,
        "ERROR": Fore.RED,
        "WARN": Fore.YELLOW,
        "WARNING": Fore.YELLOW,
        "INFO": Fore.GREEN,
        "DEBUG": Fore.BLUE,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        return color + logging.Formatter.format(self, record)


_LOGGER = logging.getLogger("token_extractor")
_LOGGER.addHandler(logging.NullHandler())


def configure_logging(level_name: str = "CRITICAL") -> None:
    """Attach the coloured stdout handler.

    The CLI calls this. Importers (the bulb dashboard) do not, so the
    library stays quiet and writes nothing to their stdout.
    """
    _LOGGER.setLevel(NAME_TO_LEVEL[level_name.upper()])
    if any(isinstance(h, logging.StreamHandler) for h in _LOGGER.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _LOGGER.addHandler(handler)


class LoginCancelled(Exception):
    """The driver asked the login to stop while it was waiting on input."""


class InteractionRequired(Exception):
    """The flow needs an answer this prompter cannot supply."""


class Prompter(ABC):
    """Everything the login flow needs from whoever is driving it.

    The CLI answers with the terminal, the bulb dashboard answers over
    HTTP. Nothing below this class calls input() or print() directly, so
    the same connectors serve both.
    """

    def note(self, text: str = "", style: str = "", tab: int = 0) -> None:
        """A line of progress. style: info | warn | error | ok | ""."""

    def entry(self, key: str, value: str, tab: int = 0) -> None:
        """One key/value line of a device listing."""
        if value:
            self.note(f"{key}: {value}", tab=tab)

    @abstractmethod
    def ask(self, key: str, label: str, secret: bool = False, hint: str = "") -> str:
        """Block until the driver supplies the answer to one question."""

    def show_image(self, kind: str, image: bytes, login_url: str = None) -> None:
        """Put a captcha or QR image (kind: "captcha" | "qr") in front of the user."""


class SilentPrompter(Prompter):
    """Answers nothing.

    For callers that supplied every credential up front and want a hard
    failure rather than a process hanging on a question nobody can see.
    """

    def ask(self, key: str, label: str, secret: bool = False, hint: str = "") -> str:
        raise InteractionRequired(f"{label} required, but this session cannot ask for it")


class ConsolePrompter(Prompter):
    """The original terminal behaviour."""

    STYLES = {"info": Fore.BLUE, "warn": Fore.YELLOW, "error": Fore.RED, "ok": Fore.GREEN}

    def __init__(self, quiet: bool = False, host: str = None):
        self.quiet = quiet
        self.host = host or "127.0.0.1"

    def note(self, text: str = "", style: str = "", tab: int = 0) -> None:
        if self.quiet:
            return
        color = self.STYLES.get(style, "")
        print(" " * tab + color + text)

    def entry(self, key: str, value: str, tab: int = 0) -> None:
        if value and not self.quiet:
            print(" " * tab + f'{Fore.YELLOW}{key + ":": <10}{Style.RESET_ALL}{value}')

    def ask(self, key: str, label: str, secret: bool = False, hint: str = "") -> str:
        self.note(f"{label} {Fore.BLUE}({hint}){Style.RESET_ALL}:" if hint else f"{label}:")
        if secret:
            return getpass("")
        return input().strip()

    def show_image(self, kind: str, image: bytes, login_url: str = None) -> None:
        if kind == "qr":
            self.note("Please scan the following QR code to log in.", "info")
        present_image_image(
            image,
            message_url=f"{'QR code' if kind == 'qr' else 'Image'} URL: {Fore.BLUE}http://{self.host}:31415",
            message_file_saved=("QR code image saved at: {}" if kind == "qr"
                                else "Captcha image saved at: {}"),
            message_manually_open_file=("Please open {} and scan the QR code." if kind == "qr"
                                        else "Please open {} and solve the captcha."),
            prompter=self,
        )
        if login_url:
            self.note()
            self.note("Alternatively you can visit the following URL:", "info")
            self.note(f"  {login_url}", "info")
            self.note()


class XiaomiCloudConnector(ABC):

    def __init__(self, prompter: Prompter = None):
        self._agent = self.generate_agent()
        self._device_id = self.generate_device_id()
        self._session = requests.session()
        self._ssecurity = None
        self.userId = None
        self._serviceToken = None
        self.prompter = prompter or SilentPrompter()
        # set by a driver that wants a login abandoned early; the QR long
        # poll is the only place that can otherwise sit for minutes
        self.cancel = threading.Event()

    @abstractmethod
    def login(self) -> bool:
        pass

    def get_homes(self, country):
        url = self.get_api_url(country) + "/v2/homeroom/gethome"
        params = {
            "data": '{"fg": true, "fetch_share": true, "fetch_share_dev": true, "limit": 300, "app_ver": 7}'}
        return self.execute_api_call_encrypted(url, params)

    def get_devices(self, country, home_id, owner_id):
        url = self.get_api_url(country) + "/v2/home/home_device_list"
        params = {
            "data": '{"home_owner": ' + str(owner_id) +
            ',"home_id": ' + str(home_id) +
            ',  "limit": 200,  "get_split_device": true, "support_smart_home": true}'
        }
        return self.execute_api_call_encrypted(url, params)

    def get_dev_cnt(self, country):
        url = self.get_api_url(country) + "/v2/user/get_device_cnt"
        params = {
            "data": '{ "fetch_own": true, "fetch_share": true}'
        }
        return self.execute_api_call_encrypted(url, params)

    def get_beaconkey(self, country, did):
        url = self.get_api_url(country) + "/v2/device/blt_get_beaconkey"
        params = {
            "data": '{"did":"' + did + '","pdid":1}'
        }
        return self.execute_api_call_encrypted(url, params)

    def execute_api_call_encrypted(self, url, params):
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": self._agent,
            "Content-Type": "application/x-www-form-urlencoded",
            "x-xiaomi-protocal-flag-cli": "PROTOCAL-HTTP2",
            "MIOT-ENCRYPT-ALGORITHM": "ENCRYPT-RC4",
        }
        cookies = {
            "userId": str(self.userId),
            "yetAnotherServiceToken": str(self._serviceToken),
            "serviceToken": str(self._serviceToken),
            "locale": "en_GB",
            "timezone": "GMT+02:00",
            "is_daylight": "1",
            "dst_offset": "3600000",
            "channel": "MI_APP_STORE"
        }
        millis = round(time.time() * 1000)
        nonce = self.generate_nonce(millis)
        signed_nonce = self.signed_nonce(nonce)
        fields = self.generate_enc_params(url, "POST", signed_nonce, nonce, params, self._ssecurity)
        response = self._session.post(url, headers=headers, cookies=cookies, params=fields)
        if response.status_code == 200:
            decoded = self.decrypt_rc4(self.signed_nonce(fields["_nonce"]), response.text)
            return json.loads(decoded)
        return None

    @staticmethod
    def get_api_url(country):
        return "https://" + ("" if country == "cn" else (country + ".")) + "api.io.mi.com/app"

    def signed_nonce(self, nonce):
        hash_object = hashlib.sha256(base64.b64decode(self._ssecurity) + base64.b64decode(nonce))
        return base64.b64encode(hash_object.digest()).decode("utf-8")

    @staticmethod
    def signed_nonce_sec(nonce, ssecurity):
        hash_object = hashlib.sha256(base64.b64decode(ssecurity) + base64.b64decode(nonce))
        return base64.b64encode(hash_object.digest()).decode("utf-8")

    @staticmethod
    def generate_nonce(millis):
        nonce_bytes = os.urandom(8) + (int(millis / 60000)).to_bytes(4, byteorder="big")
        return base64.b64encode(nonce_bytes).decode()

    @staticmethod
    def generate_agent():
        agent_id = "".join(
            map(lambda i: chr(i), [random.randint(65, 69) for _ in range(13)])
        )
        random_text = "".join(map(lambda i: chr(i), [random.randint(97, 122) for _ in range(18)]))
        return f"{random_text}-{agent_id} APP/com.xiaomi.mihome APPV/10.5.201"

    @staticmethod
    def generate_device_id():
        return "".join(map(lambda i: chr(i), [random.randint(97, 122) for _ in range(6)]))

    @staticmethod
    def generate_signature(url, signed_nonce, nonce, params):
        signature_params = [url.split("com")[1], signed_nonce, nonce]
        for k, v in params.items():
            signature_params.append(f"{k}={v}")
        signature_string = "&".join(signature_params)
        signature = hmac.new(base64.b64decode(signed_nonce), msg=signature_string.encode(), digestmod=hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()

    @staticmethod
    def generate_enc_signature(url, method, signed_nonce, params):
        signature_params = [str(method).upper(), url.split("com")[1].replace("/app/", "/")]
        for k, v in params.items():
            signature_params.append(f"{k}={v}")
        signature_params.append(signed_nonce)
        signature_string = "&".join(signature_params)
        return base64.b64encode(hashlib.sha1(signature_string.encode("utf-8")).digest()).decode()

    @staticmethod
    def generate_enc_params(url, method, signed_nonce, nonce, params, ssecurity):
        params["rc4_hash__"] = XiaomiCloudConnector.generate_enc_signature(url, method, signed_nonce, params)
        for k, v in params.items():
            params[k] = XiaomiCloudConnector.encrypt_rc4(signed_nonce, v)
        params.update({
            "signature": XiaomiCloudConnector.generate_enc_signature(url, method, signed_nonce, params),
            "ssecurity": ssecurity,
            "_nonce": nonce,
        })
        return params

    @staticmethod
    def to_json(response_text):
        return json.loads(response_text.replace("&&&START&&&", ""))

    @staticmethod
    def encrypt_rc4(password, payload):
        r = ARC4.new(base64.b64decode(password))
        r.encrypt(bytes(1024))
        return base64.b64encode(r.encrypt(payload.encode())).decode()

    @staticmethod
    def decrypt_rc4(password, payload):
        r = ARC4.new(base64.b64decode(password))
        r.encrypt(bytes(1024))
        return r.encrypt(base64.b64decode(payload))


class PasswordXiaomiCloudConnector(XiaomiCloudConnector):

    def __init__(self, prompter: Prompter = None, username: str = None, password: str = None):
        super().__init__(prompter)
        self._username = username
        self._password = password
        self._sign = None
        self._cUserId = None
        self._passToken = None
        self._location = None
        self._code = None

    def login(self) -> bool:
        if not self._username:
            self._username = self.prompter.ask(
                "username", "Username", hint="email, phone number or user ID")
        if not self._password:
            self._password = self.prompter.ask(
                "password", "Password", secret=True,
                hint="not displayed for privacy reasons")

        self.prompter.note()
        self.prompter.note("Logging in...", "info")
        self.prompter.note()

        self._session.cookies.set("sdkVersion", "accountsdk-18.8.15", domain="mi.com")
        self._session.cookies.set("sdkVersion", "accountsdk-18.8.15", domain="xiaomi.com")
        self._session.cookies.set("deviceId", self._device_id, domain="mi.com")
        self._session.cookies.set("deviceId", self._device_id, domain="xiaomi.com")

        if not self.login_step_1():
            self.prompter.note("Invalid username.", "error")
            return False

        if not self.login_step_2():
            self.prompter.note("Invalid login or password.", "error")
            return False

        if self._location and not self._serviceToken and not self.login_step_3():
            self.prompter.note("Unable to get service token.", "error")
            return False

        return True

    def login_step_1(self):
        _LOGGER.debug("login_step_1")
        url = "https://account.xiaomi.com/pass/serviceLogin?sid=xiaomiio&_json=true"
        headers = {
            "User-Agent": self._agent,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        cookies = {
            "userId": self._username
        }
        response = self._session.get(url, headers=headers, cookies=cookies)
        _LOGGER.debug(response.text)
        json_resp = self.to_json(response.text)
        if response.status_code == 200:
            if "_sign" in json_resp:
                self._sign = json_resp["_sign"]
                return True
            elif "ssecurity" in json_resp:
                self._ssecurity = json_resp["ssecurity"]
                self.userId = json_resp["userId"]
                self._cUserId = json_resp["cUserId"]
                self._passToken = json_resp["passToken"]
                self._location = json_resp["location"]
                self._code = json_resp["code"]

                return True

        return False

    def login_step_2(self) -> bool:
        _LOGGER.debug("login_step_2")
        url: str = "https://account.xiaomi.com/pass/serviceLoginAuth2"
        headers: dict = {
            "User-Agent": self._agent,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        fields: dict = {
            "sid": "xiaomiio",
            "hash": hashlib.md5(str.encode(self._password)).hexdigest().upper(),
            "callback": "https://sts.api.io.mi.com/sts",
            "qs": "%3Fsid%3Dxiaomiio%26_json%3Dtrue",
            "user": self._username,
            "_sign": self._sign,
            "_json": "true"
        }
        _LOGGER.debug("login_step_2: URL: %s", url)
        _LOGGER.debug("login_step_2: Fields: %s", fields)

        response = self._session.post(url, headers=headers, params=fields, allow_redirects=False)
        _LOGGER.debug("login_step_2: Response text: %s", response.text)

        valid: bool = response is not None and response.status_code == 200

        if valid:
            json_resp: dict = self.to_json(response.text)
            if "captchaUrl" in json_resp and json_resp["captchaUrl"] is not None:
                captcha_code: str = self.handle_captcha(json_resp["captchaUrl"])
                if not captcha_code:
                    _LOGGER.debug("Could not solve captcha.")
                    return False
                # Add captcha code to the fields and retry
                fields["captCode"] = captcha_code
                _LOGGER.debug("Retrying login with captcha.")
                response = self._session.post(url, headers=headers, params=fields, allow_redirects=False)
                _LOGGER.debug("login_step_2: Retry Response text: %s", response.text[:1000])
                if response is not None and response.status_code == 200:
                    json_resp = self.to_json(response.text)
                else:
                    _LOGGER.error("Login failed even after captcha.")
                    return False
                if "code" in json_resp and json_resp["code"] == 87001:
                    self.prompter.note("Invalid captcha.", "error")
                    return False

            valid = "ssecurity" in json_resp and len(str(json_resp["ssecurity"])) > 4
            if valid:
                self._ssecurity = json_resp["ssecurity"]
                self.userId = json_resp.get("userId", None)
                self._cUserId = json_resp.get("cUserId", None)
                self._passToken = json_resp.get("passToken", None)
                self._location = json_resp.get("location", None)
                self._code = json_resp.get("code", None)
            else:
                if "notificationUrl" in json_resp:
                    verify_url = json_resp["notificationUrl"]
                    return self.do_2fa_email_flow(verify_url)
                else:
                    _LOGGER.error("login_step_2: Login failed, server returned: %s", json_resp)
        else:
            _LOGGER.error("login_step_2: HTTP status: %s; Response: %s", response.status_code, response.text[:500])
        return valid

    def login_step_3(self):
        _LOGGER.debug("login_step_3")
        headers = {
            "User-Agent": self._agent,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = self._session.get(self._location, headers=headers)
        _LOGGER.debug(response.text)
        if response.status_code == 200:
            self._serviceToken = response.cookies.get("serviceToken")
        return response.status_code == 200

    def handle_captcha(self, captcha_url: str) -> str:
        # Full URL in case it's relative
        if captcha_url.startswith("/"):
            captcha_url = "https://account.xiaomi.com" + captcha_url

        _LOGGER.debug("Downloading captcha image from: %s", captcha_url)
        response = self._session.get(captcha_url, stream=False)
        if response.status_code != 200:
            _LOGGER.error("Unable to fetch captcha image.")
            return ""

        self.prompter.note("Captcha verification required.", "warn")
        self.prompter.show_image("captcha", response.content)

        captcha_solution: str = self.prompter.ask(
            "captcha", "Enter captcha as shown in the image", hint="case-sensitive")
        self.prompter.note()
        return captcha_solution


    def do_2fa_email_flow(self, notification_url: str) -> bool:
        """
        Handles the email-based 2FA flow and extracts ssecurity + serviceToken.
        Robust to cases where verifyEmail returns non-JSON/empty body.
        """
        # 1) Open notificationUrl (authStart)
        headers = {
            "User-Agent": self._agent,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        _LOGGER.debug("Opening notificationUrl (authStart): %s", notification_url)
        r = self._session.get(notification_url, headers=headers)
        _LOGGER.debug("authStart final URL: %s status=%s", r.url, r.status_code)

        # 2) Fetch identity options (list)
        context = parse_qs(urlparse(notification_url).query)["context"][0]
        list_params = {
            "sid": "xiaomiio",
            "context": context,
            "_locale": "en_US"
        }
        _LOGGER.debug("GET /identity/list params=%s", list_params)
        r = self._session.get("https://account.xiaomi.com/identity/list", params=list_params, headers=headers)
        _LOGGER.debug("identity/list status=%s", r.status_code)

        # 3) Request email ticket
        send_params = {
            "_dc": str(int(time.time() * 1000)),
            "sid": "xiaomiio",
            "context": list_params["context"],
            "mask": "0",
            "_locale": "en_US"
        }
        send_data = {
            "retry": "0",
            "icode": "",
            "_json": "true",
            "ick": self._session.cookies.get("ick", "")
        }
        _LOGGER.debug("sendEmailTicket POST url=https://account.xiaomi.com/identity/auth/sendEmailTicket params=%s", send_params)
        _LOGGER.debug("sendEmailTicket data=%s", send_data)
        r = self._session.post("https://account.xiaomi.com/identity/auth/sendEmailTicket",
                               params=send_params, data=send_data, headers=headers)
        try:
            jr = r.json()
        except Exception:
            jr = {}
        _LOGGER.debug("sendEmailTicket response status=%s json=%s", r.status_code, jr)

        # 4) Ask user for the email code and verify
        self.prompter.note("Two factor authentication required, please provide the code from the email.", "warn")
        self.prompter.note()
        code = self.prompter.ask("code_2fa", "2FA Code")
        self.prompter.note()

        verify_params = {
            "_flag": "8",
            "_json": "true",
            "sid": "xiaomiio",
            "context": list_params["context"],
            "mask": "0",
            "_locale": "en_US"
        }
        verify_data = {
            "_flag": "8",
            "ticket": code,
            "trust": "false",
            "_json": "true",
            "ick": self._session.cookies.get("ick", "")
        }
        r = self._session.post("https://account.xiaomi.com/identity/auth/verifyEmail",
                               params=verify_params, data=verify_data, headers=headers)
        if r.status_code != 200:
            _LOGGER.error("verifyEmail failed: status=%s body=%s", r.status_code, r.text[:500])
            return False

        try:
            jr = r.json()
            _LOGGER.debug("verifyEmail response status=%s json=%s", r.status_code, jr)
            finish_loc = jr.get("location")
        except Exception:
            # Non-JSON or empty; try to extract from headers or body
            _LOGGER.debug("verifyEmail returned non-JSON, attempting fallback extraction.")
            finish_loc = r.headers.get("Location")
            if not finish_loc and r.text:
                m = re.search(r'https://account\.xiaomi\.com/identity/result/check\?[^"\']+', r.text)
                if m:
                    finish_loc = m.group(0)

        # Fallback: directly hit result/check using existing identity_session/context
        if not finish_loc:
            _LOGGER.debug("Using fallback call to /identity/result/check")
            r0 = self._session.get(
                "https://account.xiaomi.com/identity/result/check",
                params={"sid": "xiaomiio", "context": context, "_locale": "en_US"},
                headers=headers,
                allow_redirects=False
            )
            _LOGGER.debug("result/check (fallback) status=%s hop-> %s", r0.status_code, r0.headers.get("Location"))
            if r0.status_code in (301, 302) and r0.headers.get("Location"):
                finish_loc = r0.url if "serviceLoginAuth2/end" in r0.url else r0.headers["Location"]

        if not finish_loc:
            _LOGGER.error("Unable to determine finish location after verifyEmail.")
            return False

        # First hop: GET identity/result/check (do NOT follow redirects to inspect Location)
        if "identity/result/check" in finish_loc:
            r = self._session.get(finish_loc, headers=headers, allow_redirects=False)
            _LOGGER.debug("result/check status=%s hop-> %s", r.status_code, r.headers.get("Location"))
            end_url = r.headers.get("Location")
        else:
            end_url = finish_loc

        if not end_url:
            _LOGGER.error("Could not find Auth2/end URL in finish chain.")
            return False

        # 6) Call Auth2/end WITHOUT redirects to capture 'extension-pragma' header containing ssecurity
        r = self._session.get(end_url, headers=headers, allow_redirects=False)
        _LOGGER.debug("Auth2/end status=%s", r.status_code)
        _LOGGER.debug("Auth2/end body(trunc)=%s", r.text[:200])
        # Some servers return 200 first (HTML 'Tips' page), then 302 on next call.
        if r.status_code == 200 and "Xiaomi Account - Tips" in r.text:
            r = self._session.get(end_url, headers=headers, allow_redirects=False)
            _LOGGER.debug("Auth2/end(second) status=%s", r.status_code)

        ext_prag = r.headers.get("extension-pragma")
        if ext_prag:
            try:
                ep_json = json.loads(ext_prag)
                ssec = ep_json.get("ssecurity")
                psec = ep_json.get("psecurity")
                _LOGGER.debug("extension-pragma present. ssecurity=%s psecurity=%s", ssec, psec)
                if ssec:
                    self._ssecurity = ssec
            except Exception as e:
                _LOGGER.debug("Failed to parse extension-pragma: %s", e)

        if not self._ssecurity:
            _LOGGER.error("extension-pragma header missing ssecurity; cannot continue.")
            return False

        # 7) Find STS redirect and visit it (to set serviceToken cookie)
        sts_url = r.headers.get("Location")
        if not sts_url and r.text:
            idx = r.text.find("https://sts.api.io.mi.com/sts")
            if idx != -1:
                end = r.text.find('"', idx)
                if end == -1:
                    end = idx + 300
                sts_url = r.text[idx:end]
        if not sts_url:
            _LOGGER.error("Auth2/end did not provide STS redirect.")
            return False

        r = self._session.get(sts_url, headers=headers, allow_redirects=True)
        _LOGGER.debug("STS final URL: %s status=%s", r.url, r.status_code)
        if r.status_code != 200:
            _LOGGER.error("STS did not complete: status=%s body=%s", r.status_code, r.text[:200])
            return False

        # Extract serviceToken from cookie jar
        self._serviceToken = self._session.cookies.get("serviceToken", domain=".sts.api.io.mi.com")
        found = bool(self._serviceToken)
        _LOGGER.debug("STS body (trunc)=%s", r.text[:20])
        if not found:
            _LOGGER.error("Could not parse serviceToken; cannot complete login.")
            return False
        _LOGGER.debug("STS did not return JSON; assuming 'ok' style response and relying on cookies.")
        _LOGGER.debug("extract_service_token: found=%s", found)

        # Mirror serviceToken to API domains expected by Mi Cloud
        self.install_service_token_cookies(self._serviceToken)

        # Update ids from cookies if available
        self.userId = self.userId or self._session.cookies.get("userId", domain=".xiaomi.com") or self._session.cookies.get("userId", domain=".sts.api.io.mi.com")
        self._cUserId = self._cUserId or self._session.cookies.get("cUserId", domain=".xiaomi.com") or self._session.cookies.get("cUserId", domain=".sts.api.io.mi.com")
        return True

    def install_service_token_cookies(self, token: str):
        for d in [".api.io.mi.com", ".io.mi.com", ".mi.com"]:
            self._session.cookies.set("serviceToken", token, domain=d)
            self._session.cookies.set("yetAnotherServiceToken", token, domain=d)


class QrCodeXiaomiCloudConnector(XiaomiCloudConnector):

    def __init__(self, prompter: Prompter = None):
        super().__init__(prompter)
        self._cUserId = None
        self._pass_token = None
        self._location = None
        self._qr_image_url = None
        self._login_url = None
        self._long_polling_url = None
        self._timeout = 300

    def login(self) -> bool:

        if not self.login_step_1():
            self.prompter.note("Unable to get login message.", "error")
            return False

        if not self.login_step_2():
            self.prompter.note("Unable to get login QR Image.", "error")
            return False

        if not self.login_step_3():
            self.prompter.note("Unable to login.", "error")
            return False

        if not self.login_step_4():
            self.prompter.note("Unable to get service token.", "error")
            return False

        return True

    def login_step_1(self) -> bool:
        _LOGGER.debug("login_step_1")
        url = "https://account.xiaomi.com/longPolling/loginUrl"
        data = {
            "_qrsize": "480",
            "qs": "%3Fsid%3Dxiaomiio%26_json%3Dtrue",
            "callback": "https://sts.api.io.mi.com/sts",
            "_hasLogo": "false",
            "sid": "xiaomiio",
            "serviceParam": "",
            "_locale": "en_GB",
            "_dc": str(int(time.time() * 1000))
        }

        response = self._session.get(url, params=data)
        _LOGGER.debug(response.text)

        if response.status_code == 200:
            response_data = self.to_json(response.text)
            if "qr" in response_data:
                self._qr_image_url = response_data["qr"]
                self._login_url = response_data["loginUrl"]
                self._long_polling_url = response_data["lp"]
                self._timeout = response_data["timeout"]
                return True
        return False

    def login_step_2(self) -> bool:
        _LOGGER.debug("login_step_2")
        url = self._qr_image_url
        _LOGGER.debug("login_step_2: Image URL: %s", url)

        response = self._session.get(url)

        valid: bool = response is not None and response.status_code == 200

        if valid:
            self.prompter.show_image("qr", response.content, login_url=self._login_url)
            return True
        else:
            _LOGGER.error("login_step_2: HTTP status: %s; Response: %s", response.status_code, response.text[:500])
        return False

    def login_step_3(self) -> bool:
        _LOGGER.debug("login_step_3")

        url = self._long_polling_url
        _LOGGER.debug("Long polling URL: " + url)

        start_time = time.time()
        # Start long polling
        while True:
            if self.cancel.is_set():
                raise LoginCancelled("QR login cancelled")
            try:
                response = self._session.get(url, timeout=10)
            except requests.exceptions.Timeout:
                _LOGGER.debug("Long polling timed out, retrying...")
                if time.time() - start_time > self._timeout:
                    _LOGGER.debug("Long polling timed out after {} seconds.".format(self._timeout))
                    break
                continue
            except requests.exceptions.RequestException as e:
                _LOGGER.error(f"An error occurred: {e}")
                break

            if response.status_code == 200:
                break
            else:
                _LOGGER.error("Long polling failed, retrying...")

        if response.status_code != 200:
            _LOGGER.error("Long polling failed with status code: " + str(response.status_code))
            return False

        _LOGGER.debug("Login successful!")
        _LOGGER.debug("Response data:")

        response_data = self.to_json(response.text)
        _LOGGER.debug(response_data)

        self.userId = response_data["userId"]
        self._ssecurity = response_data["ssecurity"]
        self._cUserId = response_data["cUserId"]
        self._pass_token = response_data["passToken"]
        self._location = response_data["location"]

        _LOGGER.debug("User ID: " + str(self.userId))
        _LOGGER.debug("Ssecurity: " + str(self._ssecurity))
        _LOGGER.debug("CUser ID: " + str(self._cUserId))
        _LOGGER.debug("Pass token: " + str(self._pass_token))
        _LOGGER.debug("Pass token: " + str(self._location))

        return True

    def login_step_4(self) -> bool:
        _LOGGER.debug("login_step_4")
        _LOGGER.debug("Fetching service token...")

        if not (location := self._location):
            _LOGGER.error("No location found.")
            return False

        response = self._session.get(location, headers={"content-type": "application/x-www-form-urlencoded"})
        if response.status_code != 200:
            return False

        self._serviceToken = response.cookies["serviceToken"]
        _LOGGER.debug("Service token: " + str(self._serviceToken))
        return True


def print_banner(prompter: Prompter) -> None:
    prompter.note(Fore.YELLOW + Style.BRIGHT + r"""
                               Xiaomi Cloud
___ ____ _  _ ____ _  _ ____    ____ _  _ ___ ____ ____ ____ ___ ____ ____ 
 |  |  | |_/  |___ |\ | [__     |___  \/   |  |__/ |__| |     |  |  | |__/ 
 |  |__| | \_ |___ | \| ___]    |___ _/\_  |  |  \ |  | |___  |  |__| |  \ 
""" + Style.NORMAL +
"""                                                        by Piotr Machowski 

    """)


def start_image_server(image: bytes) -> None:
    class ImgHttpHandler(BaseHTTPRequestHandler):

        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(image)

        def log_message(self, msg, *args) -> None:
            _LOGGER.debug(msg, *args)

    httpd = HTTPServer(("", 31415), ImgHttpHandler)
    _LOGGER.info("server address: %s", httpd.server_address)
    _LOGGER.info("hostname: %s", socket.gethostname())

    thread = threading.Thread(target = httpd.serve_forever)
    thread.daemon = True
    thread.start()


def present_image_image(
        image_content: bytes,
        message_url: str,
        message_file_saved: str,
        message_manually_open_file: str,
        prompter: Prompter,
) -> None:
    try:
        # Try to serve an image file
        start_image_server(image_content)
        prompter.note(message_url)
    except Exception as e1:
        _LOGGER.debug(e1)
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_content)
            tmp_path: str = tmp.name
        prompter.note(message_file_saved.format(tmp_path))
        try:
            img = Image.open(tmp_path)
            img.show()
        except Exception as e2:
            _LOGGER.debug(e2)
            prompter.note(message_manually_open_file.format(tmp_path))


def collect_devices(connector: XiaomiCloudConnector, servers: list[str],
                    prompter: Prompter = None) -> list:
    """Walk every home on every server and return the raw listing.

    Shape: [{"server", "homes": [{"home_id", "home_owner", "devices": [...]}]}].
    The device dicts are the cloud's own, so "token", "localip", "mac" and
    "model" are all in there untouched. Progress goes to the prompter, which
    is what lets the CLI and the dashboard share this loop.
    """
    prompter = prompter or SilentPrompter()
    output = []
    for current_server in servers:
        all_homes = []
        homes = connector.get_homes(current_server)
        if homes is not None:
            for h in homes["result"]["homelist"]:
                all_homes.append({"home_id": h["id"], "home_owner": connector.userId})
        dev_cnt = connector.get_dev_cnt(current_server)
        if dev_cnt is not None:
            for h in dev_cnt["result"]["share"]["share_family"]:
                all_homes.append({"home_id": h["home_id"], "home_owner": h["home_owner"]})

        if len(all_homes) == 0:
            prompter.note(f'No homes found for server "{current_server}".', "error")

        for home in all_homes:
            devices = connector.get_devices(current_server, home["home_id"], home["home_owner"])
            home["devices"] = []
            if devices is None:
                prompter.note(f"Unable to get devices from server {current_server}.", "error")
                continue
            if devices["result"]["device_info"] is None or len(devices["result"]["device_info"]) == 0:
                prompter.note(f'No devices found for server "{current_server}" @ home "{home["home_id"]}".', "error")
                continue
            prompter.note(f'Devices found for server "{current_server}" @ home "{home["home_id"]}":')
            for device in devices["result"]["device_info"]:
                device_data = {**device}
                prompter.note("---------", "info", 3)
                if "name" in device:
                    prompter.entry("NAME", device["name"], 3)
                if "did" in device:
                    prompter.entry("ID", device["did"], 3)
                    if "blt" in device["did"]:
                        beaconkey = connector.get_beaconkey(current_server, device["did"])
                        if beaconkey and "result" in beaconkey and "beaconkey" in beaconkey["result"]:
                            prompter.entry("BLE KEY", beaconkey["result"]["beaconkey"], 3)
                            device_data["BLE_DATA"] = beaconkey["result"]
                if "mac" in device:
                    prompter.entry("MAC", device["mac"], 3)
                if "localip" in device:
                    prompter.entry("IP", device["localip"], 3)
                if "token" in device:
                    prompter.entry("TOKEN", device["token"], 3)
                if "model" in device:
                    prompter.entry("MODEL", device["model"], 3)
                home["devices"].append(device_data)
            prompter.note("---------", "info", 3)
            prompter.note()
        output.append({"server": current_server, "homes": all_homes})
    return output


def flatten_devices(output: list) -> list[dict]:
    """collect_devices() output as one flat list of the fields a LAN client
    actually needs. Used by the bulb dashboard."""
    rows = []
    for server in output:
        for home in server["homes"]:
            for device in home.get("devices", []):
                rows.append({
                    "server": server["server"],
                    "home_id": home["home_id"],
                    "name": device.get("name", ""),
                    "did": device.get("did", ""),
                    "model": device.get("model", ""),
                    "mac": device.get("mac", ""),
                    "ip": device.get("localip", ""),
                    "token": device.get("token", ""),
                    "online": bool(device.get("isOnline", False)),
                })
    return rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.non_interactive and (not args.username or not args.password):
        parser.error("You need to specify username and password or run as interactive.")

    init(autoreset=True)
    configure_logging(args.log_level)
    prompter = ConsolePrompter(quiet=args.non_interactive, host=args.host)

    print_banner(prompter)
    if args.non_interactive:
        connector = PasswordXiaomiCloudConnector(prompter, args.username, args.password)
    else:
        prompter.note("Please select a way to log in:")
        prompter.note(f" p{Fore.BLUE} - using password")
        prompter.note(f" q{Fore.BLUE} - using QR code")
        log_in_method = ""
        while not log_in_method in ["P", "Q"]:
            log_in_method = input("p/q: ").upper()
        if log_in_method == "P":
            connector = PasswordXiaomiCloudConnector(prompter, args.username, args.password)
        else:
            connector = QrCodeXiaomiCloudConnector(prompter)
        prompter.note()

    try:
        logged = connector.login()
    except InteractionRequired as exc:
        parser.error(f"{exc}; rerun in interactive mode")
        return

    if logged:
        prompter.note("Logged in.", "ok")
        prompter.note()
        servers_to_check = get_servers_to_check(args, prompter)
        prompter.note()
        output = collect_devices(connector, servers_to_check, prompter)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json.dumps(output, indent=4))
    else:
        prompter.note("Unable to log in.", "error")

    if not args.non_interactive:
        prompter.note()
        prompter.note("Press ENTER to finish")
        input()


def get_servers_to_check(args, prompter: Prompter) -> list[str]:
    servers_str = ", ".join(SERVERS)
    if args.server is not None:
        server = args.server
    elif args.non_interactive:
        server = ""
    else:
        server = prompter.ask("server", "Select server",
                              hint=f"one of: {servers_str}; Leave empty to check all available")
        while server not in ["", *SERVERS]:
            prompter.note(f"Invalid server provided. Valid values: {servers_str}", "error")
            server = prompter.ask("server", "Server")

    prompter.note()
    if not server == "":
        servers_to_check = [server]
    else:
        servers_to_check = [*SERVERS]
    return servers_to_check


if __name__ == "__main__":
    main()
