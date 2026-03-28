"""
Test for chatgpt Codex model (gpt-5.3-codex) system message filtering.

This test verifies that system messages are properly filtered out when using
Codex models, which don't support system messages.
"""
import pytest
from litellm.llms.chatgpt.common_utils import get_chatgpt_default_instructions
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config


class TestOpenAIGPT5Config:
    """Test the is_model_gpt_5_codex_model function."""

    def test_is_model_gpt_5_codex_model_with_dot_version(self):
        """Test that gpt-5.3-codex is correctly detected as a Codex model."""
        assert OpenAIGPT5Config.is_model_gpt_5_codex_model("gpt-5.3-codex") is True

    def test_is_model_gpt_5_codex_model_with_dash_version(self):
        """Test that gpt-5-codex is correctly detected as a Codex model."""
        assert OpenAIGPT5Config.is_model_gpt_5_codex_model("gpt-5-codex") is True

    def test_is_model_gpt_5_codex_model_with_full_path(self):
        """Test that codex/gpt-5.3-codex is correctly detected as a Codex model."""
        assert OpenAIGPT5Config.is_model_gpt_5_codex_model("codex/gpt-5.3-codex") is True

    def test_is_model_gpt_5_codex_model_returns_false_for_non_codex(self):
        """Test that non-Codex models are not detected as Codex models."""
        assert OpenAIGPT5Config.is_model_gpt_5_codex_model("gpt-5") is False
        assert OpenAIGPT5Config.is_model_gpt_5_codex_model("gpt-4") is False
        assert OpenAIGPT5Config.is_model_gpt_5_codex_model("gpt-5-chat") is False


class TestChatGPTConfigTransformMessages:
    """Test the _transform_messages method for chat completion API."""

    def setup_method(self):
        self.config = ChatGPTConfig()

    def test_system_messages_filtered_for_codex_model(self):
        """Test that system messages are filtered out for Codex models."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = self.config._transform_messages(messages, "gpt-5.3-codex")

        # System message should be filtered out
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_system_messages_not_filtered_for_non_codex_model(self):
        """Test that system messages are kept for non-Codex models."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = self.config._transform_messages(messages, "gpt-4")

        # System message should be kept
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_multiple_messages_filtered_for_codex_model(self):
        """Test that multiple system messages are filtered out for Codex models."""
        messages = [
            {"role": "system", "content": "System message 1"},
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "System message 2"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = self.config._transform_messages(messages, "gpt-5.3-codex")

        # Both system messages should be filtered out
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


class TestChatGPTResponsesAPIConfigExtractSystemMessages:
    """Test the _extract_and_filter_system_messages method for Responses API."""

    def setup_method(self):
        self.config = ChatGPTResponsesAPIConfig()

    def test_extract_system_messages_from_message_type(self):
        """Test extracting system messages from Responses API input."""
        input_data = [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "System instruction"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        ]
        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        # System message should be extracted
        assert len(filtered_input) == 1
        assert filtered_input[0]["role"] == "user"
        assert extracted == "System instruction"

    def test_extract_easy_input_message(self):
        """Test extracting easy_input_message content."""
        input_data = [
            {"type": "easy_input_message", "content": "System instructions here"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        ]
        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        # easy_input_message should be extracted
        assert len(filtered_input) == 1
        assert filtered_input[0]["type"] == "message"
        assert extracted == "System instructions here"

    def test_extract_system_text_content(self):
        """Test extracting content with system-like text."""
        input_data = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "System: Do this"},
                    {"type": "input_text", "text": "Regular text"},
                ]
            },
        ]
        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        # System text should be extracted, regular text kept
        assert len(filtered_input[0]["content"]) == 1
        assert filtered_input[0]["content"][0]["text"] == "Regular text"
        assert extracted == "System: Do this"

    def test_string_input_passthrough(self):
        """Test that string input is passed through unchanged."""
        input_data = "Just a string input"
        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)
        assert filtered_input == input_data
        assert extracted == ""

    def test_multiple_system_messages_merged(self):
        """Test that multiple system messages are merged into instructions."""
        input_data = [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "First instruction"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "Second instruction"}]},
        ]
        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        # Both system messages should be extracted and merged
        assert len(filtered_input) == 1
        assert "First instruction\n\nSecond instruction" == extracted

    def test_no_system_content_returns_original_content(self):
        """Test that content is preserved when nothing needs extraction."""
        input_data = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello!"},
                    {"type": "input_image", "image_url": "https://example.com/image.png"},
                    "plain text",
                ],
            },
            "raw entry",
        ]

        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        assert extracted == ""
        assert filtered_input == input_data

    def test_non_dict_content_items_are_preserved_when_filtering_system_text(self):
        """Test that non-dict content survives alongside extracted system-like text."""
        input_data = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    "plain text",
                    {"type": "input_text", "text": "System: Do this first"},
                    {"type": "input_text", "text": "Regular text"},
                    123,
                ],
            }
        ]

        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        assert extracted == "System: Do this first"
        assert filtered_input == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    "plain text",
                    {"type": "input_text", "text": "Regular text"},
                    123,
                ],
            }
        ]

    def test_multiple_extracted_fragments_preserve_order(self):
        """Test that extracted fragments keep their original order."""
        input_data = [
            {"type": "easy_input_message", "content": "Top-level instruction"},
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "System: First"},
                    {"type": "input_text", "text": "Regular text"},
                    {"type": "input_text", "text": "You are a code assistant"},
                ],
            },
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "Final instruction"}]},
        ]

        filtered_input, extracted = self.config._extract_and_filter_system_messages(input_data)

        assert extracted == (
            "Top-level instruction\n\n"
            "System: First\n\n"
            "You are a code assistant\n\n"
            "Final instruction"
        )
        assert filtered_input == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Regular text"}],
            }
        ]

    def test_instructions_required_for_codex_in_transform_request(self):
        """Test that instructions are added for Codex models in transform_responses_api_request."""
        from litellm.types.router import GenericLiteLLMParams

        # Simulate the transform_responses_api_request flow for Codex
        input_data = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        ]
        result = self.config.transform_responses_api_request(
            model="gpt-5.3-codex",
            input=input_data,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Instructions should be present for Codex models
        assert "instructions" in result
        assert result["instructions"] == get_chatgpt_default_instructions()

    def test_system_messages_merged_into_instructions_for_codex(self):
        """Test that system messages are merged into instructions for Codex models."""
        from litellm.types.router import GenericLiteLLMParams

        input_data = [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are a Python expert."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        ]
        result = self.config.transform_responses_api_request(
            model="gpt-5.3-codex",
            input=input_data,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # System message should be merged into instructions
        assert "instructions" in result
        assert "You are a Python expert." in result["instructions"]
        # User message should still be in input
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"

    def test_instructions_not_added_for_non_codex_in_transform_request(self):
        """Test that default ChatGPT instructions are added for non-Codex models."""
        from litellm.types.router import GenericLiteLLMParams

        input_data = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        ]
        result = self.config.transform_responses_api_request(
            model="gpt-4",
            input=input_data,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Instructions should contain the default ChatGPT instructions for non-Codex models
        assert "instructions" in result
        assert "Codex" in result["instructions"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
