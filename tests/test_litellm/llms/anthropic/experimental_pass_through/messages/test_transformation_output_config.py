"""
Regression tests for /v1/messages passthrough param stripping.

Claude Code sub-agents (Explore, general-purpose) emit output_config.effort and
thinking blocks inherited from the parent session. When routed to a model that
doesn't support them (e.g. Haiku 4.5), the upstream Anthropic API returns 400
"This model does not support the effort parameter" and the fallback chain
swallows the request.

The transform should strip those params for models that don't support them,
while preserving them for 4.6+/4.7 adaptive-thinking models.
"""

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


def _transform(model: str, **extra) -> dict:
    cfg = AnthropicMessagesConfig()
    optional = {"max_tokens": 100, **extra}
    return cfg.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=optional,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def test_haiku_4_5_strips_output_config() -> None:
    out = _transform(
        "claude-haiku-4-5-20251001",
        output_config={"effort": "low"},
    )
    assert "output_config" not in out


def test_haiku_4_5_strips_thinking() -> None:
    out = _transform(
        "claude-haiku-4-5-20251001",
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    assert "thinking" not in out


def test_opus_4_7_keeps_output_config() -> None:
    out = _transform(
        "claude-opus-4-7",
        output_config={"effort": "high"},
    )
    assert out.get("output_config") == {"effort": "high"}


def test_sonnet_4_6_keeps_output_config() -> None:
    out = _transform(
        "claude-sonnet-4-6",
        output_config={"effort": "medium"},
    )
    assert out.get("output_config") == {"effort": "medium"}


def test_sonnet_4_6_keeps_thinking() -> None:
    out = _transform(
        "claude-sonnet-4-6",
        thinking={"type": "enabled", "budget_tokens": 4096},
    )
    assert out.get("thinking") == {"type": "enabled", "budget_tokens": 4096}


def test_prefixed_anthropic_haiku_strips_output_config() -> None:
    out = _transform(
        "anthropic/claude-haiku-4-5",
        output_config={"effort": "low"},
    )
    assert "output_config" not in out
