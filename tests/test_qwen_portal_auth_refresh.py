import json

import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.qwen_portal.authenticator import Authenticator
from litellm.llms.qwen_portal.chat.transformation import QwenPortalConfig
from litellm.llms.qwen_portal.common_utils import GetAccessTokenError


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
        assert args
        assert kwargs
        return _DummyResponse(self.payload)


def test_refresh_tokens_preserves_existing_metadata(monkeypatch, tmp_path):
    auth_file = tmp_path / "auth.qwen_portal.json"
    authenticator = Authenticator()
    authenticator.auth_file = str(auth_file)
    authenticator.token_dir = str(tmp_path)

    existing_data = {
        "refresh_token": "refresh-token",
        "resource_url": "portal.qwen.ai",
        "token_type": "Bearer",
        "custom_field": "keep-me",
    }

    monkeypatch.setattr(
        "litellm.llms.qwen_portal.authenticator._get_httpx_client",
        lambda: _DummyClient({"access_token": "fresh-token", "expires_in": 3600}),
    )

    refreshed = authenticator._refresh_tokens("refresh-token", existing_data)

    assert refreshed == {"access_token": "fresh-token"}
    written = json.loads(auth_file.read_text())
    assert written["access_token"] == "fresh-token"
    assert written["refresh_token"] == "refresh-token"
    assert written["resource_url"] == "portal.qwen.ai"
    assert written["token_type"] == "Bearer"
    assert written["custom_field"] == "keep-me"


def test_read_auth_file_uses_compat_path(monkeypatch, tmp_path):
    compat_file = tmp_path / "auth.qwen_portal.json"
    compat_file.write_text(json.dumps({"access_token": "compat-token"}))

    authenticator = Authenticator()
    authenticator.auth_file = str(tmp_path / "missing" / "auth.qwen_portal.json")
    authenticator.token_dir = str(tmp_path / "missing")

    monkeypatch.setattr(
        "litellm.llms.qwen_portal.authenticator.AUTH_FILE_COMPAT_PATH",
        str(compat_file),
    )

    auth_data = authenticator._read_auth_file()

    assert auth_data == {"access_token": "compat-token"}
    assert authenticator.auth_file == str(compat_file)
    assert authenticator.token_dir == str(tmp_path)


def test_get_access_token_raises_refresh_error_message(monkeypatch):
    authenticator = Authenticator()
    auth_data = {
        "access_token": "stale-token",
        "refresh_token": "refresh-token",
        "expires_at": 0,
    }

    monkeypatch.setattr(authenticator, "_read_auth_file", lambda: auth_data)
    monkeypatch.setattr(authenticator, "_is_token_expired", lambda _: True)
    def fake_refresh(refresh_token, existing_data=None):
        assert refresh_token == "refresh-token"
        assert existing_data is auth_data
        raise GetAccessTokenError(message="refresh failed", status_code=401)

    monkeypatch.setattr(authenticator, "_refresh_tokens", fake_refresh)

    with pytest.raises(GetAccessTokenError, match="refresh failed"):
        authenticator.get_access_token()


def test_qwen_config_wraps_get_access_token_error(monkeypatch):
    config = QwenPortalConfig()
    monkeypatch.setattr(
        config.authenticator,
        "get_access_token",
        lambda: (_ for _ in ()).throw(
            GetAccessTokenError(message="refresh failed", status_code=401)
        ),
    )

    with pytest.raises(AuthenticationError, match="refresh failed"):
        config.validate_environment(
            headers={},
            model="qwen3-vl-plus",
            messages=[],
            optional_params={},
            litellm_params={},
        )
