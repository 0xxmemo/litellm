"""
Local audio-transcription provider — OpenAI-compatible /v1/audio/transcriptions.

Compatible with any local Whisper-serving process:
- faster-whisper-server  (default: http://localhost:10300)
- whisper.cpp HTTP server (http://localhost:8080)
- LocalAI                (http://localhost:8080)
- Speecht5-server, etc.

The base URL is resolved in priority order:
  1. api_base kwarg / litellm_params
  2. LOCAL_TRANSCRIPTION_API_BASE env var
  3. http://localhost:10300  (faster-whisper-server default)
"""

from typing import Optional

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.hosted_vllm.transcriptions.transformation import (
    HostedVLLMAudioTranscriptionConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import FileTypes

_DEFAULT_BASE = "http://localhost:10300"


class LocalTranscriptionError(BaseLLMException):
    pass


class LocalTranscriptionConfig(HostedVLLMAudioTranscriptionConfig):
    """
    Transcription config for a locally-running Whisper-compatible server.

    Inherits multipart request formatting from HostedVLLMAudioTranscriptionConfig
    and overrides URL resolution and model-name stripping.
    """

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
                get_secret_str("LOCAL_TRANSCRIPTION_API_BASE")
                or _DEFAULT_BASE
            )
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/audio/transcriptions"):
            api_base = f"{api_base}/v1/audio/transcriptions"
        return api_base

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        # Strip provider prefix so the server sees only the model name
        if "/" in model:
            model = model.split("/", 1)[1]
        return AudioTranscriptionRequestData(
            data={"model": model, "file": audio_file, **optional_params},
        )
