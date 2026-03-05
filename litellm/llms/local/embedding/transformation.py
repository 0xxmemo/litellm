"""
Local embedding provider — OpenAI-compatible endpoint running on localhost.

Targets any server that exposes a /v1/embeddings endpoint without authentication:
- Ollama   (default: http://localhost:11434)
- LM Studio (http://localhost:1234)
- Infinity  (http://localhost:7997)
- FastEmbed-server, TEI, etc.

The base URL is resolved in priority order:
  1. api_base kwarg / litellm_params
  2. LOCAL_EMBEDDING_API_BASE env var
  3. OLLAMA_API_BASE env var
  4. http://localhost:11434  (Ollama default)
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.hosted_vllm.embedding.transformation import HostedVLLMEmbeddingConfig
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_DEFAULT_BASE = "http://localhost:11434"


class LocalEmbeddingError(BaseLLMException):
    pass


class LocalEmbeddingConfig(HostedVLLMEmbeddingConfig):
    """
    Embedding config for a locally-running OpenAI-compatible server.

    Inherits all request/response transformation from HostedVLLMEmbeddingConfig
    and overrides only URL resolution and auth handling.
    """

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Local servers typically need no auth; skip Authorization header
        return {"Content-Type": "application/json", **headers}

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
            api_base = (
                get_secret_str("LOCAL_EMBEDDING_API_BASE")
                or get_secret_str("OLLAMA_API_BASE")
                or _DEFAULT_BASE
            )
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/embeddings"):
            api_base = f"{api_base}/v1/embeddings" if not api_base.endswith("/v1") else f"{api_base}/embeddings"
        return api_base

    def transform_embedding_request(
        self,
        model: str,
        input: Any,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        if isinstance(input, str):
            input = [input]
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]
        return {"model": model, "input": input, **optional_params}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return LocalEmbeddingError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
