from typing import Any, List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
)
from .streaming_utils import ChatGPTToolCallNormalizer


def _chatgpt_message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "input_text" and isinstance(
                    block.get("text"), str
                ):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _merge_system_and_developer_into_instruction_text(
    messages: List[AllMessageValues],
) -> Tuple[List[AllMessageValues], str]:
    """Split system/developer turns out of `messages` and merge their text for `instructions`.

    Content is not discarded: it is concatenated (in order) and returned so the caller can
    merge it into the request's top-level ``instructions`` field. Remaining messages are the
    conversation without those roles (ChatGPT rejects them on the wire).
    """
    instruction_parts: List[str] = []
    conversation_messages: List[AllMessageValues] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            raw = _chatgpt_message_content_to_text(msg.get("content"))
            instruction_parts.append(raw)
            continue
        conversation_messages.append(msg)
    merged_instructions = "\n\n".join(
        part for part in instruction_parts if part.strip() != ""
    )
    return conversation_messages, merged_instructions


class ChatGPTConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = self.authenticator.get_api_base()
        try:
            dynamic_api_key = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            api_key or "", account_id, session_id
        )
        return {**default_headers, **validated_headers}

    def post_stream_processing(self, stream: Any) -> Any:
        return ChatGPTToolCallNormalizer(stream)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        optional_params.setdefault("stream", False)
        return optional_params

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        return super()._transform_messages(messages, model)

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Merge system/developer content into top-level ``instructions`` (Responses-style)."""
        conversation_messages, merged_from_roles = (
            _merge_system_and_developer_into_instruction_text(messages)
        )
        optional_params = dict(optional_params)
        existing_instructions = (optional_params.get("instructions") or "").strip()
        if merged_from_roles:
            optional_params["instructions"] = (
                f"{merged_from_roles}\n\n{existing_instructions}"
                if existing_instructions
                else merged_from_roles
            )
        elif not existing_instructions:
            optional_params["instructions"] = get_chatgpt_default_instructions()
        messages_transformed = self._transform_messages(
            messages=conversation_messages, model=model
        )
        optional_params.pop("max_retries", None)
        return {
            "model": model,
            "messages": messages_transformed,
            **optional_params,
        }
