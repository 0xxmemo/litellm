"""
KimiCodeConfig — Kimi Code provider using Anthropic Messages API format.

Transformation flow (OpenAI legacy → Anthropic):
──────────────────────────────────────────────────
  INPUT:  OpenAI /v1/chat/completions format (standard litellm client input)
  OUTPUT: Anthropic /v1/messages format (sent to Kimi Code API)

  1. Client sends OpenAI-format request → litellm.completion()
  2. AnthropicConfig.transform_request() converts to Anthropic Messages format:
     - messages[]              → messages[] (role/content restructured)
     - system message          → top-level "system" parameter
     - tools/functions         → Anthropic tool format
     - max_tokens, temperature → passed through
     - response_format         → Anthropic structured output
  3. Request sent to Kimi API at /coding/v1/messages
  4. AnthropicConfig.transform_response() converts Anthropic response → OpenAI:
     - content[]               → choices[].message.content
     - tool_use blocks         → choices[].message.tool_calls[]
     - stop_reason             → choices[].finish_reason
     - usage                   → usage (prompt/completion tokens)
  5. Client receives standard OpenAI-format ModelResponse

Authentication:
  OAuth device-code flow via auth.kimi.com. Bearer token injected in
  validate_environment() as the Authorization header (Anthropic convention).

Agent identification:
  X-Msh-Platform, X-Msh-Version, X-Msh-Device-Id, etc. headers are
  injected to identify the coding agent to the Kimi API.

Extends AnthropicConfig (not OpenAIConfig) because the Kimi Code API
accepts the Anthropic Messages format at its /coding/v1/messages endpoint.
"""

from typing import Dict, List, Optional

from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError, get_kimi_code_default_headers


class KimiCodeConfig(AnthropicConfig):
    """
    Kimi Code provider — receives OpenAI format, outputs Anthropic format.

    Inherits full OpenAI↔Anthropic translation from AnthropicConfig:
      - transform_request()   : OpenAI messages → Anthropic messages
      - transform_response()  : Anthropic response → OpenAI ModelResponse
      - map_openai_params()   : OpenAI params → Anthropic params
      - get_error_class()     : error handling

    Overrides for Kimi-specific behaviour:
      - validate_environment(): OAuth Bearer auth + agent ID headers
      - get_complete_url()    : Kimi API endpoint (api.kimi.com/coding/v1/messages)
      - custom_llm_provider   : "kimi_code"
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
        """
        Build request headers for the Kimi Code API.

        Sets three header groups:
          1. Auth — Authorization: Bearer <oauth_token>
          2. Anthropic compat — anthropic-version, content-type
          3. Agent ID — User-Agent, X-Msh-Platform, X-Msh-Version, etc.
        """
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
                "anthropic-version": "2023-06-01",
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
        """
        Build the Kimi Code API URL for Anthropic Messages format.

        Default: https://api.kimi.com/coding/v1/messages
        Override via KIMI_CODE_API_BASE env var or api_base model param.
        """
        base = api_base or self.authenticator.get_api_base()

        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        if base.endswith("/v1/"):
            return f"{base}messages"
        if base.endswith("/"):
            return f"{base}v1/messages"
        return f"{base}/v1/messages"
