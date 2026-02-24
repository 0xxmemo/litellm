"""
Alibaba Anthropic-compatible transformation config.
"""

from typing import Optional

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str


class AlibabaMessagesConfig(AnthropicMessagesConfig):
    """
    Alibaba Anthropic-compatible configuration.
    Endpoint:
    - https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "alibaba"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or get_secret_str("ALIBABA_API_KEY")
            or get_secret_str("DASHSCOPE_API_KEY")
            or litellm.api_key
        )

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        return (
            api_base
            or get_secret_str("ALIBABA_ANTHROPIC_API_BASE")
            or "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages"
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
        base_url = self.get_api_base(api_base=api_base)
        if base_url.endswith("/v1/messages"):
            return base_url
        if base_url.endswith("/"):
            return f"{base_url}v1/messages"
        return f"{base_url}/v1/messages"
