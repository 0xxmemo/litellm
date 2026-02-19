"""
QwenPortalConfig — routes requests through the Qwen Portal OpenAI-compatible API
using OAuth tokens obtained via device-code flow.

Follows the same pattern as ChatGPTConfig: extends OpenAIConfig and overrides
_get_openai_compatible_provider_info() to inject the OAuth Bearer token as api_key.
"""

from typing import List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError


class QwenPortalConfig(OpenAIConfig):
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
        except GetAccessTokenError:
            # Allow deployment creation even when auth is unavailable
            # (e.g. expired tokens at proxy startup). Actual auth is
            # resolved lazily in validate_environment at request time.
            dynamic_api_key = "qwen-portal-pending-auth"
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
        try:
            fresh_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="qwen_portal",
                message=str(e),
            )
        return super().validate_environment(
            headers, model, messages, optional_params, litellm_params,
            fresh_token, api_base,
        )

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
        return optional_params
