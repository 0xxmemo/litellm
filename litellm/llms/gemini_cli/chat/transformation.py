"""
Gemini CLI OAuth provider config.

Extends GoogleAIStudioGeminiConfig to inherit all Gemini-native request/response
transformation methods. The actual HTTP call with OAuth Bearer auth is handled
by handler.py — this class provides the config interface that LiteLLM's
pre-processing expects (param mapping, message transformation, etc.).

Usage: model = "gemini_cli/<model-name>"
e.g.  model = "gemini_cli/gemini-2.5-flash-lite"
"""

from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig


class GeminiCLIConfig(GoogleAIStudioGeminiConfig):
    """
    Config for the gemini_cli provider.

    Inherits all Gemini-native transformation from GoogleAIStudioGeminiConfig.
    Registered in _lazy_imports_registry and ProviderConfigManager so LiteLLM
    can use it for pre-processing (message role translation, param mapping, etc.).
    The completion HTTP call goes through handler.py with OAuth Bearer auth.
    """

    pass

