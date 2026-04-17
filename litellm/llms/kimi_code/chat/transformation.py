"""
KimiCodeConfig — Kimi Code provider (OpenAI-compatible).

The Kimi Code API at api.kimi.com/coding/v1 speaks OpenAI format natively.
This config handles OAuth authentication and agent identification headers.

Routed through base_llm_http_handler (not the OpenAI SDK client) because
Kimi's API requires custom agent identification headers (X-Msh-Platform,
User-Agent, etc.) that the SDK client doesn't forward.

When clients connect via the proxy's /v1/messages endpoint (Anthropic format),
LiteLLM's built-in adapter translates Anthropic↔OpenAI automatically — no
provider-level transformation needed.

Authentication:
  OAuth device-code flow via auth.kimi.com. Bearer token injected in
  validate_environment() as the Authorization header.

Agent identification:
  X-Msh-Platform, X-Msh-Version, X-Msh-Device-Id, etc. headers are
  injected in validate_environment().

Tool names:
  Kimi rejects function names that contain characters outside
  ``[a-zA-Z0-9_-]`` (e.g. dots in ``plugin.tool``). We rewrite outgoing
  names and restore them on responses / streaming chunks.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import httpx

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator, OpenAIConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError, get_kimi_code_default_headers


_KIMI_TOOL_NAME_VALID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def _is_valid_kimi_tool_function_name(name: str) -> bool:
    return bool(name and _KIMI_TOOL_NAME_VALID.fullmatch(name))


def _sanitize_kimi_tool_function_name(name: str) -> str:
    """Map arbitrary tool names to Kimi-allowed identifiers."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "tool"
    if not s[0].isalpha():
        s = "t_" + s
    return s


def _build_kimi_tool_name_maps(names: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Forward: original -> kimi-safe; reverse: kimi-safe -> original."""
    forward: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    used: set[str] = set()
    for raw in names:
        if raw in forward:
            continue
        if _is_valid_kimi_tool_function_name(raw):
            safe = raw
        else:
            base = _sanitize_kimi_tool_function_name(raw)
            safe = base
            n = 0
            while safe in used:
                n += 1
                safe = f"{base}_{n}"
        used.add(safe)
        forward[raw] = safe
        reverse[safe] = raw
    return forward, reverse


def _collect_tool_names_from_request_payload(data: dict) -> List[str]:
    out: List[str] = []

    def add(n: Any) -> None:
        if isinstance(n, str) and n not in out:
            out.append(n)

    tools = data.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            if isinstance(fn, dict):
                add(fn.get("name"))
            # legacy / kimi builtin shapes
            if isinstance(t.get("name"), str):
                add(t.get("name"))

    for fn in data.get("functions") or []:
        if isinstance(fn, dict):
            add(fn.get("name"))

    messages = data.get("messages") or []
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool" and isinstance(m.get("name"), str):
                add(m.get("name"))
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                if tc.get("type") == "function":
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        add(fn.get("name"))

    tc = data.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function")
        if isinstance(fn, dict):
            add(fn.get("name"))

    return out


def _apply_tool_forward_map_to_payload(data: dict, fmap: Dict[str, str]) -> None:
    if not fmap:
        return

    def map_name(n: Optional[str]) -> Optional[str]:
        if n is None:
            return None
        return fmap.get(n, n)

    tools = data.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            if isinstance(fn, dict) and "name" in fn:
                fn["name"] = map_name(fn.get("name"))
            if "name" in t and isinstance(t.get("name"), str):
                t["name"] = map_name(t.get("name"))

    for fn in data.get("functions") or []:
        if isinstance(fn, dict) and "name" in fn:
            fn["name"] = map_name(fn.get("name"))

    messages = data.get("messages") or []
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool" and isinstance(m.get("name"), str):
                m["name"] = map_name(m.get("name"))
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                if tc.get("type") == "function":
                    fn = tc.get("function")
                    if isinstance(fn, dict) and "name" in fn:
                        fn["name"] = map_name(fn.get("name"))

    tc = data.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function")
        if isinstance(fn, dict) and "name" in fn:
            fn["name"] = map_name(fn.get("name"))


def _restore_tool_names_in_response_obj(obj: Any, rmap: Dict[str, str]) -> None:
    """Restore original tool names in a chat completion JSON (non-streaming)."""
    if not rmap or not isinstance(obj, dict):
        return
    for ch in obj.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message")
        if isinstance(msg, dict):
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                    fn = tc["function"]
                    n = fn.get("name")
                    if isinstance(n, str) and n in rmap:
                        fn["name"] = rmap[n]


class _KimiToolNameStreamIterator(OpenAIChatCompletionResponseIterator):
    """Rewrites tool function names in streaming chunks back to client names."""

    def __init__(
        self,
        *,
        streaming_response: Union[Any, Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
        reverse_map: Dict[str, str],
    ):
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        self._reverse_map = reverse_map

    def chunk_parser(self, chunk: dict) -> Any:
        if self._reverse_map:
            chunk = copy.deepcopy(chunk)
            for ch in chunk.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta")
                if isinstance(delta, dict):
                    for tc in delta.get("tool_calls") or []:
                        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                            fn = tc["function"]
                            n = fn.get("name")
                            if isinstance(n, str) and n in self._reverse_map:
                                fn["name"] = self._reverse_map[n]
        return super().chunk_parser(chunk=chunk)


class KimiCodeConfig(OpenAIConfig):
    """
    Kimi Code provider — OpenAI format with OAuth auth + agent headers.

    Extends OpenAIConfig for request/response format.
    Uses base_llm_http_handler (not OpenAI SDK) for full header control.

    Overrides:
      - validate_environment(): OAuth Bearer token + agent ID headers
      - get_complete_url(): Kimi endpoint at /coding/v1/chat/completions
      - custom_llm_provider: "kimi_code"
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()
        self._kimi_tool_reverse_map: Dict[str, str] = {}

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "kimi_code"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        try:
            access_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="kimi_code",
                message=str(e),
            )

        headers.update(
            {
                "authorization": f"Bearer {access_token}",
                "content-type": "application/json",
            }
        )
        headers.update(get_kimi_code_default_headers())
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Kimi Code requires an explicit ``thinking`` object on chat completions.

        If omitted, the API responds with HTTP 400 and a misleading error:
        ``invalid temperature: only 0.6 is allowed for this model`` (see
        kosong/kimi-cli: requests set ``thinking`` via extra_body). Default to
        disabled reasoning to match non-thinking clients; callers may pass
        ``thinking`` / map reasoning to enable it.
        """
        request = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if request.get("thinking") is None:
            request["thinking"] = {"type": "disabled"}

        names = _collect_tool_names_from_request_payload(request)
        _fmap, rmap = _build_kimi_tool_name_maps(names)
        self._kimi_tool_reverse_map = rmap
        if _fmap:
            _apply_tool_forward_map_to_payload(request, _fmap)

        return request

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers
        response_json = raw_response.json()
        rmap = getattr(self, "_kimi_tool_reverse_map", None) or {}
        if rmap:
            response_json = copy.deepcopy(response_json)
            _restore_tool_names_in_response_obj(response_json, rmap)
        final_response_obj = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=response_json,
                model_response_object=model_response,
                hidden_params={"headers": raw_response.headers},
                _response_headers=dict(raw_response.headers),
            ),
        )
        return final_response_obj

    def get_model_response_iterator(
        self,
        streaming_response: Union[Any, Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        rmap = getattr(self, "_kimi_tool_reverse_map", None) or {}
        if rmap:
            return _KimiToolNameStreamIterator(
                streaming_response=streaming_response,
                sync_stream=sync_stream,
                json_mode=json_mode,
                reverse_map=rmap,
            )
        return super().get_model_response_iterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = api_base or self.authenticator.get_api_base()

        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        if base.endswith("/v1/"):
            return f"{base}chat/completions"
        if base.endswith("/"):
            return f"{base}v1/chat/completions"
        return f"{base}/v1/chat/completions"
