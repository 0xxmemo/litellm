"""
KimiCodeConfig — routes requests through the Kimi Code OpenAI-compatible API
using OAuth tokens obtained via device-code flow (kimi login).

Follows the same pattern as ChatGPTConfig / QwenPortalConfig: extends
OpenAIConfig and overrides _get_openai_compatible_provider_info() to inject
the OAuth Bearer token as api_key.

Agent identification headers (User-Agent, X-Msh-*) are injected as
extra_headers in main.py (same pattern as github_copilot).
"""

from typing import Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError


class KimiCodeConfig(OpenAIConfig):
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
                llm_provider="kimi_code",
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider
