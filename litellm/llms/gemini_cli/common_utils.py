"""
Constants and helpers for Gemini CLI subscription OAuth.

Mirrors the ChatGPT provider pattern but targets Google's Code Assist API
using OAuth credentials extracted from the installed Gemini CLI
(npm: @anthropic-ai/gemini-cli).
"""

import os
import platform
from typing import Any, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Google OAuth2 endpoints
GEMINI_CLI_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GEMINI_CLI_TOKEN_URL = "https://oauth2.googleapis.com/token"
GEMINI_CLI_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"

# Code Assist API
GEMINI_CLI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# OAuth scopes required by Gemini CLI
GEMINI_CLI_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform "
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile"
)

# Code Assist project discovery
GEMINI_CLI_CODE_ASSIST_URL = "https://cloudcode-pa.googleapis.com"


class GeminiCLIAuthError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetAccessTokenError(GeminiCLIAuthError):
    pass


class RefreshAccessTokenError(GeminiCLIAuthError):
    pass


def get_gemini_cli_default_headers(access_token: str) -> dict:
    """Build default headers for Gemini API requests."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    return headers
