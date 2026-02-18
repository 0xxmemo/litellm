"""
Qwen Portal OAuth authenticator — reads cached tokens from disk,
refreshes via chat.qwen.ai token endpoint, and provides Bearer tokens
for the OpenAI-compatible Qwen Portal API.

Follows OpenClaw's approach for Qwen device-code OAuth.
"""

import json
import os
import time
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    QWEN_CLIENT_ID,
    QWEN_DEFAULT_API_BASE,
    QWEN_TOKEN_URL,
    QWEN_USER_AGENT,
    GetAccessTokenError,
    RefreshAccessTokenError,
)

TOKEN_EXPIRY_SKEW_SECONDS = 60


class Authenticator:
    """Manages Qwen Portal OAuth tokens persisted in a JSON file."""

    def __init__(self) -> None:
        self.token_dir = os.getenv(
            "QWEN_PORTAL_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/qwen_portal"),
        )
        self.auth_file = os.path.join(
            self.token_dir,
            os.getenv("QWEN_PORTAL_AUTH_FILE", "auth.qwen_portal.json"),
        )
        self._ensure_token_dir()

    def get_api_base(self) -> str:
        env_base = os.getenv("QWEN_PORTAL_API_BASE")
        if env_base:
            return env_base

        auth_data = self._read_auth_file()
        if auth_data:
            resource_url = auth_data.get("resource_url")
            if resource_url:
                return self._normalize_url(resource_url)

        return QWEN_DEFAULT_API_BASE

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not url.startswith("http"):
            url = f"https://{url}"
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url

    def get_access_token(self) -> str:
        auth_data = self._read_auth_file()
        if auth_data:
            access_token = auth_data.get("access_token")
            if access_token and not self._is_token_expired(auth_data):
                return access_token
            refresh_token = auth_data.get("refresh_token")
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token, auth_data)
                    return refreshed["access_token"]
                except RefreshAccessTokenError as exc:
                    verbose_logger.warning(
                        "Qwen Portal refresh token failed, re-login required: %s",
                        exc,
                    )

        # Fall back to syncing from Qwen Code CLI's credential file
        synced = self._try_sync_qwen_cli_creds()
        if synced:
            return synced

        raise GetAccessTokenError(
            message=(
                "Qwen Portal OAuth tokens are missing or expired. "
                "Run 'litellmctl auth qwen' to authenticate, or "
                "refresh with 'litellmctl auth refresh qwen'."
            ),
            status_code=401,
        )

    def _try_sync_qwen_cli_creds(self) -> Optional[str]:
        """Sync credentials from Qwen Code CLI's ~/.qwen/oauth_creds.json."""
        qwen_cli_path = os.path.expanduser("~/.qwen/oauth_creds.json")
        try:
            with open(qwen_cli_path, "r") as f:
                cli_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

        access_token = cli_data.get("access_token")
        expiry_date = cli_data.get("expiry_date")
        if not access_token or not expiry_date:
            return None

        # expiry_date is in milliseconds in the Qwen CLI format
        expires_at = expiry_date / 1000.0 if expiry_date > 1e12 else expiry_date
        if time.time() >= expires_at - TOKEN_EXPIRY_SKEW_SECONDS:
            refresh_token = cli_data.get("refresh_token")
            if not refresh_token:
                return None
            try:
                refreshed = self._refresh_tokens(refresh_token, {
                    "resource_url": cli_data.get("resource_url"),
                })
                return refreshed["access_token"]
            except RefreshAccessTokenError:
                return None

        # Token is still valid — cache it locally
        auth_record = {
            "access_token": access_token,
            "refresh_token": cli_data.get("refresh_token"),
            "expires_at": int(expires_at),
            "resource_url": cli_data.get("resource_url"),
            "token_type": cli_data.get("token_type", "Bearer"),
        }
        self._write_auth_file(auth_record)
        return access_token

    def _ensure_token_dir(self) -> None:
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _read_auth_file(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.auth_file, "r") as f:
                return json.load(f)
        except IOError:
            return None
        except json.JSONDecodeError as exc:
            verbose_logger.warning("Invalid Qwen Portal auth file: %s", exc)
            return None

    def _write_auth_file(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.auth_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as exc:
            verbose_logger.error("Failed to write Qwen Portal auth file: %s", exc)

    def _is_token_expired(self, auth_data: Dict[str, Any]) -> bool:
        expires_at = auth_data.get("expires_at")
        if expires_at is None:
            return True
        return time.time() >= float(expires_at) - TOKEN_EXPIRY_SKEW_SECONDS

    def _refresh_tokens(
        self, refresh_token: str, existing_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            resp = client.post(
                QWEN_TOKEN_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": QWEN_USER_AGENT,
                },
                content=(
                    f"grant_type=refresh_token"
                    f"&refresh_token={refresh_token}"
                    f"&client_id={QWEN_CLIENT_ID}"
                ),
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=f"Qwen Portal refresh failed: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise RefreshAccessTokenError(
                message=f"Qwen Portal refresh failed: {exc}",
                status_code=400,
            )

        access_token = data.get("access_token")
        if not access_token:
            raise RefreshAccessTokenError(
                message=f"Qwen Portal refresh response missing access_token: {data}",
                status_code=400,
            )

        expires_in = data.get("expires_in", 3600)
        resource_url = data.get("resource_url")
        if not resource_url and existing_data:
            resource_url = existing_data.get("resource_url")

        auth_record = {
            "access_token": access_token,
            "refresh_token": data.get("refresh_token", refresh_token),
            "expires_at": int(time.time() + expires_in),
            "resource_url": resource_url,
            "token_type": data.get("token_type", "Bearer"),
        }
        self._write_auth_file(auth_record)
        return {"access_token": access_token}
