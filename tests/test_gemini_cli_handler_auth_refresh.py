from types import SimpleNamespace

import httpx
import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.gemini_cli.handler import GeminiCLIHandler, _async_make_call, _sync_make_call
from litellm.llms.gemini_cli.common_utils import GetAccessTokenError
from litellm.llms.vertex_ai.common_utils import VertexAIError


class _MockLogging:
    def __init__(self):
        self.optional_params = {}
        self.litellm_call_id = "test-call-id"

    def update_environment_variables(self, model=None, optional_params=None, litellm_params=None):
        self.optional_params = optional_params or {}

    def pre_call(self, input=None, api_key=None, additional_args=None):
        pass

    def post_call(self, input=None, api_key=None, original_response=None, additional_args=None):
        pass


class _MockResponse:
    def __init__(self, status_code: int, text: str, headers: dict = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "https://cloudcode-pa.googleapis.com")
            resp = httpx.Response(self.status_code, request=req, text=self.text)
            raise httpx.HTTPStatusError("error", request=req, response=resp)

    def json(self):
        # Code Assist envelope expected by _UnwrappingResponse
        return {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "ok"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            }
        }


class _MockClient401Then200:
    def __init__(self):
        self.calls = 0

    def post(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _MockResponse(401, "unauthorized")
        return _MockResponse(200, "ok")


class _MockClientAlways401:
    def __init__(self):
        self.calls = 0

    def post(self, **kwargs):
        self.calls += 1
        return _MockResponse(401, "unauthorized")


class _MockStreamResponse(_MockResponse):
    def read(self):
        return self.text.encode()

    def iter_lines(self):
        return iter(["data: {\"response\": {\"candidates\": []}}"])


class _MockStreamClient401Then200:
    def __init__(self):
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _MockStreamResponse(401, "unauthorized")
        return _MockStreamResponse(200, "ok")


class _MockAsyncStreamResponse(_MockResponse):
    async def aread(self):
        return self.text.encode()

    async def aiter_lines(self):
        yield "data: {\"response\": {\"candidates\": []}}"


class _MockAsyncStreamClient401Then200:
    def __init__(self):
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _MockAsyncStreamResponse(401, "unauthorized")
        return _MockAsyncStreamResponse(200, "ok")


def test_sync_completion_retries_once_on_401_and_uses_refreshed_token(monkeypatch):
    handler = GeminiCLIHandler()

    # Stub auth state + refresh flow
    monkeypatch.setattr(
        handler.authenticator,
        "get_access_token",
        lambda: "stale_access_token",
    )
    monkeypatch.setattr(
        handler.authenticator,
        "get_project_id",
        lambda: "test-project",
    )
    monkeypatch.setattr(
        handler.authenticator,
        "_read_auth_file",
        lambda: {"refresh_token": "refresh-token"},
    )
    monkeypatch.setattr(
        handler.authenticator,
        "_refresh_tokens",
        lambda rt, auth_data=None: {
            "access_token": "fresh_access_token",
            "refresh_token": rt,
        },
    )

    # Avoid depending on full Gemini request builder internals
    monkeypatch.setattr(
        "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
        lambda **kwargs: {"contents": []},
    )

    # Keep this test focused on retry logic, not Gemini response shape conversion
    monkeypatch.setattr(
        "litellm.llms.gemini.chat.transformation.GoogleAIStudioGeminiConfig.transform_response",
        lambda self, **kwargs: {"ok": True},
    )

    model_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    client = _MockClient401Then200()

    result = handler.completion(
        model="gemini-3-pro-preview",
        messages=[{"role": "user", "content": "hi"}],
        model_response=model_response,
        print_verbose=lambda *args, **kwargs: None,
        encoding=None,
        logging_obj=_MockLogging(),
        optional_params={},
        litellm_params={},
        acompletion=False,
        timeout=30,
        extra_headers=None,
        client=client,
        api_base=None,
        logger_fn=None,
    )

    # Should have retried exactly once after 401
    assert client.calls == 2
    assert result == {"ok": True}


def test_sync_completion_raises_when_401_and_no_refresh_token(monkeypatch):
    handler = GeminiCLIHandler()

    monkeypatch.setattr(handler.authenticator, "get_access_token", lambda: "stale")
    monkeypatch.setattr(handler.authenticator, "get_project_id", lambda: "test-project")
    monkeypatch.setattr(handler.authenticator, "_read_auth_file", lambda: {})
    monkeypatch.setattr(
        "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
        lambda **kwargs: {"contents": []},
    )

    model_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    client = _MockClientAlways401()

    try:
        handler.completion(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": "hi"}],
            model_response=model_response,
            print_verbose=lambda *args, **kwargs: None,
            encoding=None,
            logging_obj=_MockLogging(),
            optional_params={},
            litellm_params={},
            acompletion=False,
            timeout=30,
            extra_headers=None,
            client=client,
            api_base=None,
            logger_fn=None,
        )
        assert False, "Expected VertexAIError"
    except VertexAIError as exc:
        assert exc.status_code == 401


def test_get_access_token_refresh_uses_stored_oauth_credentials(monkeypatch):
    handler = GeminiCLIHandler()

    auth_data = {
        "refresh_token": "refresh-token",
        "oauth_client_id": "stored-client-id",
        "oauth_client_secret": "stored-client-secret",
    }
    monkeypatch.setattr(handler.authenticator, "_read_auth_file", lambda: auth_data)
    monkeypatch.setattr(handler.authenticator, "_is_token_expired", lambda _: True)

    captured = {}

    def fake_refresh(refresh_token, auth_data=None):
        captured["refresh_token"] = refresh_token
        captured["auth_data"] = auth_data
        return {"access_token": "fresh_access_token"}

    monkeypatch.setattr(handler.authenticator, "_refresh_tokens", fake_refresh)

    assert handler.authenticator.get_access_token() == "fresh_access_token"
    assert captured == {
        "refresh_token": "refresh-token",
        "auth_data": auth_data,
    }


def test_resolve_oauth_credentials_prefers_auth_file_over_cached_or_env(monkeypatch):
    handler = GeminiCLIHandler()
    handler.authenticator._client_id = "cached-client-id"
    handler.authenticator._client_secret = "cached-client-secret"

    monkeypatch.setenv("GEMINI_CLI_OAUTH_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GEMINI_CLI_OAUTH_CLIENT_SECRET", "env-client-secret")

    auth_data = {
        "oauth_client_id": "file-client-id",
        "oauth_client_secret": "file-client-secret",
    }

    client_id, client_secret = handler.authenticator._resolve_oauth_credentials(
        auth_data=auth_data
    )

    assert client_id == "file-client-id"
    assert client_secret == "file-client-secret"


def test_async_streaming_retries_once_on_401_and_uses_refreshed_token(monkeypatch):
    import asyncio

    handler = GeminiCLIHandler()
    client = _MockAsyncStreamClient401Then200()
    logging_obj = _MockLogging()

    monkeypatch.setattr(
        handler.authenticator,
        "_read_auth_file",
        lambda: {"refresh_token": "refresh-token"},
    )
    monkeypatch.setattr(
        handler.authenticator,
        "_refresh_tokens",
        lambda rt, auth_data=None: {
            "access_token": "fresh_access_token",
            "refresh_token": rt,
        },
    )

    stream = asyncio.run(
        _async_make_call(
            client=client,
            api_base="https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
            headers={"Authorization": "Bearer stale_access_token"},
            data="{}",
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": "hi"}],
            logging_obj=logging_obj,
            authenticator=handler.authenticator,
        )
    )

    assert client.calls == 2
    assert stream is not None


def test_sync_streaming_retries_once_on_401_and_uses_refreshed_token(monkeypatch):
    handler = GeminiCLIHandler()
    client = _MockStreamClient401Then200()
    logging_obj = _MockLogging()

    monkeypatch.setattr(
        handler.authenticator,
        "_read_auth_file",
        lambda: {"refresh_token": "refresh-token"},
    )
    monkeypatch.setattr(
        handler.authenticator,
        "_refresh_tokens",
        lambda rt, auth_data=None: {
            "access_token": "fresh_access_token",
            "refresh_token": rt,
        },
    )

    stream = _sync_make_call(
        client=client,
        api_base="https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        headers={"Authorization": "Bearer stale_access_token"},
        data="{}",
        model="gemini-3-pro-preview",
        messages=[{"role": "user", "content": "hi"}],
        logging_obj=logging_obj,
        authenticator=handler.authenticator,
    )

    assert client.calls == 2
    assert stream is not None


def test_sync_completion_raises_auth_error_when_refresh_credentials_missing(monkeypatch):
    handler = GeminiCLIHandler()

    monkeypatch.setattr(handler.authenticator, "get_access_token", lambda: "stale")
    monkeypatch.setattr(handler.authenticator, "get_project_id", lambda: "test-project")
    monkeypatch.setattr(
        handler.authenticator,
        "_read_auth_file",
        lambda: {"refresh_token": "refresh-token"},
    )

    def fake_refresh(refresh_token, auth_data=None):
        raise GetAccessTokenError(
            message="Could not resolve Gemini CLI OAuth credentials.",
            status_code=401,
        )

    monkeypatch.setattr(handler.authenticator, "_refresh_tokens", fake_refresh)
    monkeypatch.setattr(
        "litellm.llms.vertex_ai.gemini.transformation._transform_request_body",
        lambda **kwargs: {"contents": []},
    )

    model_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    with pytest.raises(AuthenticationError, match="Could not resolve Gemini CLI OAuth credentials"):
        handler.completion(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": "hi"}],
            model_response=model_response,
            print_verbose=lambda *args, **kwargs: None,
            encoding=None,
            logging_obj=_MockLogging(),
            optional_params={},
            litellm_params={},
            acompletion=False,
            timeout=30,
            extra_headers=None,
            client=_MockClientAlways401(),
            api_base=None,
            logger_fn=None,
        )
