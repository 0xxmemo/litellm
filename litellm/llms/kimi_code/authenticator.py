"""
Kimi Code OAuth authenticator — reads cached tokens from disk,
refreshes via auth.kimi.com token endpoint, and provides Bearer tokens
for the OpenAI-compatible Kimi Code API.

Follows the same pattern as QwenPortalConfig / ChatGPTConfig.
"""

import json
import os
import time
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    KIMI_CODE_CLIENT_ID,
    KIMI_CODE_DEFAULT_API_BASE,
    KIMI_CODE_TOKEN_URL,
    KIMI_CODE_USER_AGENT,
    GetAccessTokenError,
    RefreshAccessTokenError,
    get_kimi_code_default_headers,
)

TOKEN_EXPIRY_SKEW_SECONDS = 60


class Authenticator:
    """Manages Kimi Code OAuth tokens persisted in a JSON file."""

    def __init__(self) -> None:
        self.token_dir = os.getenv(
            "KIMI_CODE_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/kimi_code"),
        )
        self.auth_file = os.path.join(
            self.token_dir,
            os.getenv("KIMI_CODE_AUTH_FILE", "auth.kimi_code.json"),
        )
        self._ensure_token_dir()

    def get_api_base(self) -> str:
        return os.getenv("KIMI_CODE_API_BASE") or KIMI_CODE_DEFAULT_API_BASE

    def get_access_token(self) -> str:
        auth_data = self._read_auth_file()
        if auth_data:
            access_token = auth_data.get("access_token")
            if access_token and not self._is_token_expired(auth_data):
                return access_token
            refresh_token = auth_data.get("refresh_token")
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token)
                    return refreshed["access_token"]
                except RefreshAccessTokenError as exc:
                    verbose_logger.warning(
                        "Kimi Code refresh token failed, re-login required: %s",
                        exc,
                    )

        synced = self._try_sync_kimi_cli_creds()
        if synced:
            return synced

        raise GetAccessTokenError(
            message=(
                "Kimi Code OAuth tokens are missing or expired. "
                "Run 'kimi login' to authenticate."
            ),
            status_code=401,
        )

    def _try_sync_kimi_cli_creds(self) -> Optional[str]:
        """Sync credentials from kimi-cli's ~/.kimi/credentials/kimi-code.json."""
        kimi_cli_path = os.path.expanduser("~/.kimi/credentials/kimi-code.json")
        try:
            with open(kimi_cli_path, "r") as f:
                cli_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

        access_token = cli_data.get("access_token")
        expires_at = cli_data.get("expires_at")
        if not access_token or not expires_at:
            return None

        if time.time() >= float(expires_at) - TOKEN_EXPIRY_SKEW_SECONDS:
            refresh_token = cli_data.get("refresh_token")
            if not refresh_token:
                return None
            try:
                refreshed = self._refresh_tokens(refresh_token)
                return refreshed["access_token"]
            except RefreshAccessTokenError:
                return None

        auth_record = {
            "access_token": access_token,
            "refresh_token": cli_data.get("refresh_token"),
            "expires_at": float(expires_at),
            "scope": cli_data.get("scope", "kimi-code"),
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
            verbose_logger.warning("Invalid Kimi Code auth file: %s", exc)
            return None

    def _write_auth_file(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.auth_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as exc:
            verbose_logger.error("Failed to write Kimi Code auth file: %s", exc)

    def _write_kimi_cli_creds(self, data: Dict[str, Any]) -> None:
        """Write back to kimi-cli credential file so both stay in sync."""
        kimi_cli_path = os.path.expanduser("~/.kimi/credentials/kimi-code.json")
        cli_record = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_at": data["expires_at"],
            "scope": data.get("scope", "kimi-code"),
            "token_type": data.get("token_type", "Bearer"),
        }
        try:
            os.makedirs(os.path.dirname(kimi_cli_path), exist_ok=True)
            with open(kimi_cli_path, "w") as f:
                json.dump(cli_record, f)
            os.chmod(kimi_cli_path, 0o600)
        except OSError as exc:
            verbose_logger.debug("Could not sync to kimi-cli creds: %s", exc)

    def _is_token_expired(self, auth_data: Dict[str, Any]) -> bool:
        expires_at = auth_data.get("expires_at")
        if expires_at is None:
            return True
        return time.time() >= float(expires_at) - TOKEN_EXPIRY_SKEW_SECONDS

    def _refresh_tokens(self, refresh_token: str) -> Dict[str, str]:
        oauth_host = os.getenv("KIMI_CODE_OAUTH_HOST", "https://auth.kimi.com")
        token_url = f"{oauth_host}/api/oauth/token"
        common_headers = get_kimi_code_default_headers()
        try:
            client = _get_httpx_client()
            resp = client.post(
                token_url,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    **common_headers,
                },
                content=(
                    f"grant_type=refresh_token"
                    f"&refresh_token={refresh_token}"
                    f"&client_id={KIMI_CODE_CLIENT_ID}"
                ),
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=f"Kimi Code refresh failed: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise RefreshAccessTokenError(
                message=f"Kimi Code refresh failed: {exc}",
                status_code=400,
            )

        access_token = data.get("access_token")
        if not access_token:
            raise RefreshAccessTokenError(
                message=f"Kimi Code refresh response missing access_token: {data}",
                status_code=400,
            )

        expires_in = data.get("expires_in", 900)
        auth_record = {
            "access_token": access_token,
            "refresh_token": data.get("refresh_token", refresh_token),
            "expires_at": time.time() + float(expires_in),
            "scope": data.get("scope", "kimi-code"),
            "token_type": data.get("token_type", "Bearer"),
        }
        self._write_auth_file(auth_record)
        self._write_kimi_cli_creds(auth_record)
        return {"access_token": access_token}
