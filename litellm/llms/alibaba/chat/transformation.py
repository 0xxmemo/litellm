"""
Translates from OpenAI's `/v1/chat/completions` to Alibaba Cloud's coding endpoint.

Handles reasoning (enable_thinking / thinking_budget), tool-calling guards for
thinking mode, web search, code interpreter, and default timeouts/limits.
"""

import re
from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

_THINKING_BUDGET_MAP = {
    "low": 2_000,
    "medium": 5_000,
    "high": 10_000,
}
_CODER_PATTERN = re.compile(r"coder", re.IGNORECASE)
_DEFAULT_THINKING_MODELS = re.compile(r"qwen3\.5", re.IGNORECASE)
_SEARCH_OPTION_KEYS = {
    "forced_search",
    "search_strategy",
    "enable_source",
    "enable_search_extension",
}

ALIBABA_DEFAULT_TIMEOUT = 600
ALIBABA_DEFAULT_STREAM_TIMEOUT = 600
ALIBABA_DEFAULT_MAX_TOKENS = 16384


class AlibabaChatConfig(OpenAIGPTConfig):
    """
    OpenAI-compatible config for Alibaba Cloud Coding Plan endpoint.
    """

    def get_supported_openai_params(self, model: str) -> list:
        params = super().get_supported_openai_params(model)
        params.extend(
            [
                "thinking",
                "reasoning_effort",
                "enable_search",
                "search_options",
                "enable_code_interpreter",
                "web_search_options",
            ]
        )
        return params

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
        extra_body = optional_params.setdefault("extra_body", {})
        if not isinstance(extra_body, dict):
            extra_body = {}
            optional_params["extra_body"] = extra_body

        is_coder_model = bool(_CODER_PATTERN.search(model))
        has_default_thinking = bool(_DEFAULT_THINKING_MODELS.search(model))
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        user_disabled_thinking = False
        if (
            isinstance(thinking_value, dict)
            and thinking_value.get("type") == "disabled"
        ):
            user_disabled_thinking = True
        elif reasoning_effort == "none":
            user_disabled_thinking = True

        if is_coder_model:
            pass
        elif user_disabled_thinking:
            extra_body["enable_thinking"] = False
        elif (
            isinstance(thinking_value, dict)
            and thinking_value.get("type") == "enabled"
        ):
            extra_body["enable_thinking"] = True
            budget = thinking_value.get("budget_tokens")
            if isinstance(budget, int) and budget > 0:
                extra_body["thinking_budget"] = budget
        elif reasoning_effort is not None:
            extra_body["enable_thinking"] = True
            extra_body["thinking_budget"] = _THINKING_BUDGET_MAP.get(
                reasoning_effort, _THINKING_BUDGET_MAP["medium"]
            )

        thinking_active = extra_body.get(
            "enable_thinking", has_default_thinking
        ) and not is_coder_model

        web_search_options = optional_params.pop("web_search_options", None)
        if isinstance(web_search_options, dict):
            extra_body["enable_search"] = True
            search_options = {}
            for key in _SEARCH_OPTION_KEYS:
                value = web_search_options.get(key)
                if value is not None:
                    search_options[key] = value
            if search_options:
                extra_body["search_options"] = search_options

        if "enable_search" in optional_params:
            extra_body["enable_search"] = optional_params.pop("enable_search")
        if "search_options" in optional_params:
            search_options = optional_params.pop("search_options")
            if isinstance(search_options, dict):
                extra_body["search_options"] = search_options
        if "enable_code_interpreter" in optional_params:
            extra_body["enable_code_interpreter"] = optional_params.pop(
                "enable_code_interpreter"
            )

        # DashScope rejects all tool_choice values except "auto" and "none"
        # when thinking mode is active.
        if thinking_active:
            tool_choice = optional_params.get("tool_choice")
            if tool_choice is not None and tool_choice not in ("auto", "none"):
                optional_params.pop("tool_choice", None)

        if "max_tokens" not in optional_params and "max_completion_tokens" not in optional_params:
            optional_params["max_tokens"] = ALIBABA_DEFAULT_MAX_TOKENS

        return optional_params

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        return super()._transform_messages(
            messages=messages, model=model, is_async=False
        )

    def get_provider_default_timeout(self) -> Optional[float]:
        return ALIBABA_DEFAULT_TIMEOUT

    def get_provider_default_stream_timeout(self) -> Optional[float]:
        return ALIBABA_DEFAULT_STREAM_TIMEOUT

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        dynamic_api_base = (
            api_base
            or get_secret_str("ALIBABA_API_BASE")
            or get_secret_str("DASHSCOPE_API_BASE")
            or "https://coding-intl.dashscope.aliyuncs.com/v1"
        )
        dynamic_api_key = (
            api_key
            or get_secret_str("ALIBABA_API_KEY")
            or get_secret_str("DASHSCOPE_API_KEY")
        )
        return dynamic_api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            api_base = "https://coding-intl.dashscope.aliyuncs.com/v1"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
