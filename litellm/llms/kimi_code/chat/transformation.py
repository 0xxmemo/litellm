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
"""

from typing import Dict, List, Optional

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError, get_kimi_code_default_headers


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
