from typing import List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    get_gemini_cli_default_headers,
)


class GeminiCLIConfig(GoogleAIStudioGeminiConfig):
    """
    Gemini CLI OAuth provider config.

    Uses OAuth tokens from the Gemini CLI (google-gemini-cli-auth style)
    to authenticate against Google's Gemini API, bypassing API key auth.

    Usage: model = "gemini_cli/<model-name>"
    e.g.  model = "gemini_cli/gemini-2.5-pro"
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = api_base or self.authenticator.get_api_base()
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

        if api_key:
            oauth_headers = get_gemini_cli_default_headers(api_key)
            return {**validated_headers, **oauth_headers}

        return validated_headers
