import json
from typing import Any, List, Optional, Union

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
)


def _ingest_system_message_content_for_instructions(
    content: Any, append_system_content: Any
) -> None:
    """Append system/developer message content to the instruction-merge buffer (preserve all text)."""
    if isinstance(content, list):
        for content_item in content:
            if isinstance(content_item, dict):
                if content_item.get("type") == "input_text":
                    text = content_item.get("text")
                    if text:
                        append_system_content(text)
                elif isinstance(content_item.get("text"), str):
                    append_system_content(content_item["text"])
            elif isinstance(content_item, str):
                append_system_content(content_item)
    elif isinstance(content, str):
        append_system_content(content)
    elif content is not None:
        append_system_content(str(content))


class ChatGPTResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        try:
            access_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token, account_id, session_id
        )
        return {**default_headers, **headers}

    def _extract_and_filter_system_messages(
        self, input: Union[str, List[Any]]
    ) -> tuple[Union[str, List[Any]], str]:
        """Pull system/developer (and related) content from `input` for merging into ``instructions``.

        This does **not** drop that content: it is accumulated into ``extracted_system_content``
        so ``transform_responses_api_request`` can concatenate it with existing instructions.
        The returned list omits those input items only because their text now lives in
        ``instructions`` (ChatGPT disallows those roles in ``input``).

        Returns:
            tuple: (input_without_merged_roles, text_to_merge_into_instructions)

        Supported shapes include:
        - ``message`` / chat-shaped items with role ``system`` or ``developer``
        - ``easy_input_message`` (treated as instruction-like payload)
        """
        if isinstance(input, str) or not isinstance(input, list):
            return input, ""

        filtered_input = []
        append_filtered_input = filtered_input.append
        system_contents: List[str] = []
        append_system_content = system_contents.append

        excluded_roles_types = frozenset(
            {"function_call", "function_call_output", "reasoning"}
        )

        for raw_item in input:
            item: Any = raw_item
            if hasattr(item, "model_dump") and callable(item.model_dump):
                try:
                    item = item.model_dump(exclude_none=True)
                except Exception:
                    pass
            if not isinstance(item, dict):
                append_filtered_input(raw_item)
                continue

            item_type = item.get("type")
            if item_type == "easy_input_message":
                content = item.get("content")
                if content:
                    append_system_content(str(content))
                continue

            content = item.get("content")
            role_lower = (item.get("role") or "").lower()
            # Chat-completions-shaped turns forwarded as Responses `input` (e.g. proxy maps
            # `messages` -> `input`) use role + content only — no top-level `type`.
            # ChatGPT rejects system/developer turns unless merged into `instructions`.
            if role_lower in ("system", "developer") and item_type not in excluded_roles_types:
                _ingest_system_message_content_for_instructions(
                    content, append_system_content
                )
                continue

            if not isinstance(content, list):
                append_filtered_input(item)
                continue

            filtered_content: Optional[List[Any]] = None
            for index, content_item in enumerate(content):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "input_text"
                    and isinstance(content_item.get("text"), str)
                ):
                    text = content_item["text"]
                    if text.startswith("System:") or text.startswith("You are"):
                        append_system_content(text)
                        if filtered_content is None:
                            filtered_content = content[:index]
                        continue

                if filtered_content is not None:
                    filtered_content.append(content_item)

            if filtered_content is not None:
                item["content"] = filtered_content
            append_filtered_input(item)

        return filtered_input, "\n\n".join(system_contents)

    def transform_responses_api_request(
        self,
        model: str,
        input: Any,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        # Move system/developer text out of `input` and merge into `instructions` (never discard)
        input, extracted_system_content = self._extract_and_filter_system_messages(input)

        request = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )

        # Concatenate extracted system/developer text with any existing instructions (preserve all)
        existing_instructions = request.get("instructions", "")
        if extracted_system_content:
            if existing_instructions:
                request["instructions"] = f"{extracted_system_content}\n\n{existing_instructions}"
            else:
                request["instructions"] = extracted_system_content
        elif not existing_instructions:
            request["instructions"] = get_chatgpt_default_instructions()
        request["store"] = False
        request["stream"] = True
        include = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        allowed_keys = {
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "include",
            "tools",
            "tool_choice",
            "reasoning",
            "previous_response_id",
            "truncation",
        }

        return {k: v for k, v in request.items() if k in allowed_keys}

    def transform_response_api_response(
        self,
        model: str,
        raw_response: Any,
        logging_obj: Any,
    ):
        content_type = (raw_response.headers or {}).get("content-type", "")
        body_text = raw_response.text or ""
        if "text/event-stream" not in content_type.lower():
            trimmed_body = body_text.lstrip()
            if not (
                trimmed_body.startswith("event:")
                or trimmed_body.startswith("data:")
                or "\nevent:" in body_text
                or "\ndata:" in body_text
            ):
                return super().transform_response_api_response(
                    model=model,
                    raw_response=raw_response,
                    logging_obj=logging_obj,
                )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        completed_response = None
        error_message = None
        for chunk in body_text.splitlines():
            stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if not stripped_chunk:
                continue
            stripped_chunk = stripped_chunk.strip()
            if not stripped_chunk:
                continue
            if stripped_chunk == STREAM_SSE_DONE_STRING:
                break
            try:
                parsed_chunk = json.loads(stripped_chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_chunk, dict):
                continue
            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                response_payload = parsed_chunk.get("response")
                if isinstance(response_payload, dict):
                    response_payload = dict(response_payload)
                    if "created_at" in response_payload:
                        response_payload["created_at"] = _safe_convert_created_field(
                            response_payload["created_at"]
                        )
                    try:
                        completed_response = ResponsesAPIResponse(**response_payload)
                    except Exception:
                        completed_response = ResponsesAPIResponse.model_construct(
                            **response_payload
                        )
                break
            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                error_obj = parsed_chunk.get("error") or (
                    parsed_chunk.get("response") or {}
                ).get("error")
                if error_obj is not None:
                    if isinstance(error_obj, dict):
                        error_message = error_obj.get("message") or str(error_obj)
                    else:
                        error_message = str(error_obj)

        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers
        return completed_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False
