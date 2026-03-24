import json

import pytest

from litellm.llms.kimi_code.authenticator import Authenticator
from litellm.llms.kimi_code.common_utils import GetAccessTokenError


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DummyClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def post(self, *args, **kwargs):
        _ = args, kwargs
        return _DummyResponse(self.payload)


def test_get_access_token_reads_legacy_auth_file(monkeypatch, tmp_path):
    home = tmp_path / "home"
    legacy_file = home / ".litellm" / "auth.kimi_code.json"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(
        json.dumps(
            {
                "access_token": "legacy-token",
                "refresh_token": "legacy-refresh",
                "expires_at": 9999999999,
                "scope": "kimi-code",
                "token_type": "Bearer",
            }
        )
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_TOKEN_DIR", str(home / ".config" / "litellm" / "kimi_code"))
    monkeypatch.setattr(
        "litellm.llms.kimi_code.authenticator.AUTH_FILE_COMPAT_PATH",
        str(legacy_file),
    )

    authenticator = Authenticator()

    assert authenticator.get_access_token() == "legacy-token"
    assert authenticator.auth_file == str(legacy_file)
    assert authenticator.token_dir == str(legacy_file.parent)


def test_get_access_token_refreshes_legacy_auth_file(monkeypatch, tmp_path):
    home = tmp_path / "home"
    legacy_file = home / ".litellm" / "auth.kimi_code.json"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(
        json.dumps(
            {
                "access_token": "stale-token",
                "refresh_token": "refresh-token",
                "expires_at": 0,
                "scope": "kimi-code",
                "token_type": "Bearer",
            }
        )
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_TOKEN_DIR", str(home / ".config" / "litellm" / "kimi_code"))
    monkeypatch.setattr(
        "litellm.llms.kimi_code.authenticator.AUTH_FILE_COMPAT_PATH",
        str(legacy_file),
    )
    monkeypatch.setattr(
        "litellm.llms.kimi_code.authenticator._get_httpx_client",
        lambda: _DummyClient({"access_token": "fresh-token", "refresh_token": "new-refresh", "expires_in": 3600}),
    )

    authenticator = Authenticator()

    assert authenticator.get_access_token() == "fresh-token"

    refreshed = json.loads(legacy_file.read_text())
    assert refreshed["access_token"] == "fresh-token"
    assert refreshed["refresh_token"] == "new-refresh"
    assert refreshed["expires_at"] > 0

    cli_file = home / ".kimi" / "credentials" / "kimi-code.json"
    cli_data = json.loads(cli_file.read_text())
    assert cli_data["access_token"] == "fresh-token"
    assert cli_data["refresh_token"] == "new-refresh"


def test_get_access_token_raises_when_credentials_missing(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_TOKEN_DIR", str(home / ".config" / "litellm" / "kimi_code"))
    monkeypatch.setattr(
        "litellm.llms.kimi_code.authenticator.AUTH_FILE_COMPAT_PATH",
        str(home / ".litellm" / "auth.kimi_code.json"),
    )

    authenticator = Authenticator()

    with pytest.raises(GetAccessTokenError) as exc_info:
        authenticator.get_access_token()

    assert exc_info.value.status_code == 401
    assert "missing or expired" in str(exc_info.value)
