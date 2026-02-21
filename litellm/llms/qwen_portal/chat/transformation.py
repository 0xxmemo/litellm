"""
QwenPortalConfig — Qwen Portal provider (OpenAI-compatible).

The Qwen Portal API at portal.qwen.ai/v1 speaks OpenAI format natively.
This config handles OAuth authentication via device-code flow.

Routed through base_llm_http_handler (not the OpenAI SDK client) so that
validate_environment() is called on every request, ensuring OAuth tokens
are refreshed before they expire.

When clients connect via the proxy's /v1/messages endpoint (Anthropic format),
LiteLLM's built-in adapter translates Anthropic<>OpenAI automatically.

Authentication:
  OAuth device-code flow via chat.qwen.ai. Bearer token injected in
  validate_environment() as the Authorization header.
  Can also sync credentials from Qwen Code CLI (~/.qwen/oauth_creds.json).
"""

from typing import Dict, List, Optional

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError


class QwenPortalConfig(OpenAIConfig):
    """
    Qwen Portal provider — OpenAI format with OAuth auth.

    Extends OpenAIConfig for request/response format.
    Uses base_llm_http_handler (not OpenAI SDK) for reliable OAuth refresh.

    Overrides:
      - validate_environment(): OAuth Bearer token on every request
      - get_complete_url(): Qwen Portal endpoint at /v1/chat/completions
      - custom_llm_provider: "qwen_portal"
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "qwen_portal"

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
                llm_provider="qwen_portal",
                message=str(e),
            )

        headers.update(
            {
                "authorization": f"Bearer {access_token}",
                "content-type": "application/json",
            }
        )
        return headers

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
