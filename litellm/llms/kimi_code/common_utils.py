"""
Constants and helpers for Kimi Code OAuth (device-code flow via auth.kimi.com).

Mirrors header generation from kimi-cli (kimi_cli.auth.oauth._common_headers
and kimi_cli.llm._kimi_default_headers) so the Kimi API accepts requests
from this proxy as a recognised coding agent.
"""

import os
import platform
import socket
from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

KIMI_CODE_OAUTH_HOST = "https://auth.kimi.com"
KIMI_CODE_TOKEN_URL = f"{KIMI_CODE_OAUTH_HOST}/api/oauth/token"
KIMI_CODE_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_CODE_DEFAULT_API_BASE = "https://api.kimi.com/coding/v1"
KIMI_CODE_VERSION = "1.12.0"
KIMI_CODE_USER_AGENT = f"KimiCLI/{KIMI_CODE_VERSION}"


class KimiCodeAuthError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetAccessTokenError(KimiCodeAuthError):
    pass


class RefreshAccessTokenError(KimiCodeAuthError):
    pass


def _ascii_header_value(value: str, *, fallback: str = "unknown") -> str:
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        sanitized = value.encode("ascii", errors="ignore").decode("ascii").strip()
        return sanitized or fallback


def _get_device_id() -> str:
    device_id_path = os.path.expanduser("~/.kimi/device_id")
    try:
        with open(device_id_path, "r") as f:
            return f.read().strip()
    except IOError:
        return ""


def get_kimi_code_default_headers() -> dict:
    """Headers required by the Kimi Code API to identify the coding agent.

    Reproduces the exact set from kimi-cli's _common_headers() + User-Agent.
    """
    device_id = _get_device_id()
    device_name = platform.node() or socket.gethostname()
    device_model = f"{platform.system()} {platform.release()} {platform.machine()}"
    raw = {
        "User-Agent": KIMI_CODE_USER_AGENT,
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": KIMI_CODE_VERSION,
        "X-Msh-Device-Id": device_id,
        "X-Msh-Device-Name": device_name,
        "X-Msh-Device-Model": device_model,
        "X-Msh-Os-Version": platform.version(),
    }
    return {k: _ascii_header_value(v) for k, v in raw.items()}
