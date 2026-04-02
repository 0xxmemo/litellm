from typing import Any, List, Optional, Tuple, cast

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
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


def _prepend_text_to_message_content(prefix: str, content: Any) -> Any:
    if not prefix:
        return content
    if isinstance(content, str):
        return f"{prefix}\n\n{content}" if content else prefix
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    return f"{prefix}\n\n{_chatgpt_message_content_to_text(content)}"


def _fold_system_and_developer_into_messages(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    """ChatGPT rejects role=system/developer on chat completions — fold into the next user/assistant turn."""
    pending: List[str] = []
    out: List[AllMessageValues] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            pending.append(_chatgpt_message_content_to_text(msg.get("content")))
            continue
        if pending and role in ("user", "assistant"):
            prefix = "\n\n".join(p for p in pending if p)
            pending.clear()
            new_msg = dict(msg)
            new_msg["content"] = _prepend_text_to_message_content(
                prefix, msg.get("content")
            )
            out.append(cast(AllMessageValues, new_msg))
            continue
        out.append(msg)
    if pending:
        prefix = "\n\n".join(p for p in pending if p)
        if prefix:
            out.insert(0, {"role": "user", "content": prefix})
    return out


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
        """Fold system/developer into user/assistant content; ChatGPT disallows those roles."""
        folded = _fold_system_and_developer_into_messages(messages)
        return super()._transform_messages(folded, model)
