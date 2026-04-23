"""
Regression test for /v1/messages fallback path dropping output_config.

When the Anthropic primary call fails and the router falls back to an
OpenAI-compat provider (codex/gpt-5.4-mini, dashscope, azure), we must not
pass output_config through as an OpenAI kwarg — upstream providers interpret
the leaked field as an auth or schema violation.

Cherry-picks upstream BerriAI/litellm db9914287a.
"""

from unittest.mock import MagicMock, patch


def test_output_config_not_forwarded_to_openai_fallback() -> None:
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value=MagicMock()) as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="azure/o1",
                api_key="test-api-key",
                output_config={"effort": "medium"},
            )
        except (ValueError, TypeError, AttributeError):
            pass

        mock_completion.assert_called_once()
        assert "output_config" not in mock_completion.call_args.kwargs
