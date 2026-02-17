"""
Self-contained handler for Gemini CLI OAuth provider.

Routes requests through Google's Code Assist proxy
(cloudcode-pa.googleapis.com) using OAuth Bearer tokens.
This is the same endpoint the actual Gemini CLI uses.

Reuses LiteLLM's existing Gemini request/response transformers
for body construction without modifying them.
"""

import json
import uuid
from functools import partial
from typing import Callable, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
    VertexAIError,
    VertexGeminiConfig,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from .authenticator import Authenticator
from .common_utils import GetAccessTokenError

CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
CODE_ASSIST_API_VERSION = "v1internal"


def _build_url(stream: bool) -> str:
    """Build Code Assist API URL for generateContent."""
    base = "{}/{}".format(CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION)
    if stream:
        return "{}:streamGenerateContent?alt=sse".format(base)
    return "{}:generateContent".format(base)


def _build_headers(access_token: str, extra_headers: Optional[dict] = None) -> dict:
    """Build request headers with OAuth Bearer token."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(access_token),
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _wrap_request(model: str, project_id: Optional[str], request_body: dict) -> dict:
    """
    Wrap the standard Gemini request body in Code Assist format.

    The Code Assist API expects:
    {
        "model": "<model-name>",
        "project": "<project-id>",
        "request": { <standard generateContent body> }
    }

    Note: the model field must NOT have a "models/" prefix — the Code Assist
    API expects the bare model name (e.g. "gemini-2.5-flash-lite").
    """
    bare_model = model.removeprefix("models/")
    wrapped = {
        "model": bare_model,
        "request": request_body,
    }
    if project_id:
        wrapped["project"] = project_id
    return wrapped


def _unwrap_response(response_data: dict) -> dict:
    """
    Unwrap Code Assist response to standard Gemini format.

    Code Assist returns: {"response": {<standard response>}, "traceId": "..."}
    """
    if "response" in response_data:
        return response_data["response"]
    return response_data


def _get_access_token(authenticator: Authenticator, model: str) -> str:
    """Get OAuth access token, raising AuthenticationError on failure."""
    try:
        return authenticator.get_access_token()
    except GetAccessTokenError as e:
        raise litellm.AuthenticationError(
            model=model,
            llm_provider="gemini_cli",
            message=str(e),
        )


class _UnwrappingResponse:
    """Wraps an httpx.Response to unwrap Code Assist envelope on .json()."""

    def __init__(self, raw_response):
        self._raw = raw_response

    def json(self):
        data = self._raw.json()
        return _unwrap_response(data)

    def __getattr__(self, name):
        return getattr(self._raw, name)


class _UnwrappingStreamIterator:
    """
    Wraps a streaming line iterator to unwrap Code Assist response envelope.

    Each SSE data line contains a JSON object with the Code Assist wrapper.
    We unwrap {"response": {...}} to just {...} for ModelResponseIterator.
    """

    def __init__(self, line_iter, sync: bool):
        self._iter = line_iter
        self._sync = sync

    def __iter__(self):
        for line in self._iter:
            yield self._process_line(line)

    async def __aiter__(self):
        async for line in self._iter:
            yield self._process_line(line)

    def _process_line(self, line: str) -> str:
        stripped = line.strip() if isinstance(line, str) else line
        if not stripped:
            return line
        # SSE format: "data: {json}"
        text = stripped
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        if text.startswith("data: "):
            json_str = text[6:]
            try:
                data = json.loads(json_str)
                unwrapped = _unwrap_response(data)
                return "data: " + json.dumps(unwrapped)
            except (json.JSONDecodeError, TypeError):
                pass
        return line


async def _async_make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: LiteLLMLoggingObj,
):
    """Async streaming call."""
    if client is None:
        client = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

    response = await client.post(
        api_base, headers=headers, data=data, stream=True, logging_obj=logging_obj
    )

    if response.status_code != 200:
        raise VertexAIError(
            status_code=response.status_code,
            message=str(await response.aread()),
            headers=dict(response.headers),
        )

    completion_stream = ModelResponseIterator(
        streaming_response=_UnwrappingStreamIterator(
            response.aiter_lines(), sync=False
        ),
        sync_stream=False,
        logging_obj=logging_obj,
    )

    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def _sync_make_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: LiteLLMLoggingObj,
):
    """Sync streaming call."""
    if client is None:
        client = HTTPHandler()

    response = client.post(
        api_base, headers=headers, data=data, stream=True, logging_obj=logging_obj
    )

    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=str(response.read()),
            headers=response.headers,
        )

    completion_stream = ModelResponseIterator(
        streaming_response=_UnwrappingStreamIterator(
            response.iter_lines(), sync=True
        ),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class GeminiCLIHandler:
    """
    Self-contained handler for gemini_cli provider.

    Reuses LiteLLM's native Gemini transformers for request/response
    formatting, wraps them in the Code Assist API envelope, and handles
    OAuth Bearer auth independently.
    """

    def __init__(self) -> None:
        self.authenticator = Authenticator()

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_base: Optional[str] = None,
        logger_fn=None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        stream: Optional[bool] = optional_params.pop("stream", None)

        # 1. Auth
        access_token = _get_access_token(self.authenticator, model)
        project_id = self.authenticator.get_project_id()

        # 2. URL
        url = api_base or _build_url(stream=bool(stream))

        # 3. Headers
        headers = _build_headers(access_token, extra_headers)

        # 4. Transform request body using existing Gemini transformers
        from litellm.llms.vertex_ai.gemini.transformation import (
            _transform_request_body,
        )

        gemini_body = _transform_request_body(
            messages=messages,
            model=model,
            optional_params=optional_params,
            custom_llm_provider="gemini",
            litellm_params=litellm_params,
            cached_content=optional_params.pop("cached_content", None),
        )

        # 5. Wrap in Code Assist envelope
        data = _wrap_request(model, project_id, gemini_body)

        # 6. Logging
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        # 7. Route: async streaming / async completion / sync streaming / sync completion
        if acompletion:
            if stream is True:
                data_str = json.dumps(data)
                return CustomStreamWrapper(
                    completion_stream=None,
                    make_call=partial(
                        _async_make_call,
                        client=(
                            client
                            if isinstance(client, AsyncHTTPHandler)
                            else None
                        ),
                        api_base=url,
                        headers=headers,
                        data=data_str,
                        model=model,
                        messages=messages,
                        logging_obj=logging_obj,
                    ),
                    model=model,
                    custom_llm_provider="vertex_ai_beta",
                    logging_obj=logging_obj,
                )

            return self._async_completion(
                model=model,
                messages=messages,
                data=data,
                url=url,
                headers=headers,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                timeout=timeout,
                client=client,
            )

        # Sync streaming
        if stream is True:
            data_str = json.dumps(data)
            return CustomStreamWrapper(
                completion_stream=None,
                make_call=partial(
                    _sync_make_call,
                    client=(
                        client
                        if isinstance(client, HTTPHandler)
                        else None
                    ),
                    api_base=url,
                    headers=headers,
                    data=data_str,
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                ),
                model=model,
                custom_llm_provider="vertex_ai_beta",
                logging_obj=logging_obj,
            )

        # Sync completion
        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, (float, int)):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            sync_client = _get_httpx_client(params=_params)
        else:
            sync_client = client

        try:
            response = sync_client.post(
                url=url, headers=headers, json=data, logging_obj=logging_obj
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise VertexAIError(
                status_code=err.response.status_code,
                message=err.response.text,
                headers=err.response.headers,
            )
        except httpx.TimeoutException:
            raise VertexAIError(
                status_code=408,
                message="Timeout error occurred.",
                headers=None,
            )

        return VertexGeminiConfig().transform_response(
            model=model,
            raw_response=_UnwrappingResponse(response),
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key="",
            request_data=data,
            messages=messages,
            encoding=encoding,
        )

    async def _async_completion(
        self,
        model: str,
        messages: list,
        data: dict,
        url: str,
        headers: dict,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        encoding,
        timeout: Optional[Union[float, httpx.Timeout]],
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ) -> ModelResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, (float, int)):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            async_client = AsyncHTTPHandler(**_params)
        else:
            async_client = client

        try:
            response = await async_client.post(
                url=url, headers=headers, json=data, logging_obj=logging_obj
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise VertexAIError(
                status_code=err.response.status_code,
                message=err.response.text,
                headers=err.response.headers,
            )
        except httpx.TimeoutException:
            raise VertexAIError(
                status_code=408,
                message="Timeout error occurred.",
                headers=None,
            )

        return VertexGeminiConfig().transform_response(
            model=model,
            raw_response=_UnwrappingResponse(response),
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key="",
            request_data=data,
            messages=messages,
            encoding=encoding,
        )
