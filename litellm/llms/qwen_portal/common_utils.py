"""
Constants and helpers for Qwen Portal OAuth (device-code flow via chat.qwen.ai).
"""

from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

QWEN_OAUTH_BASE = "https://chat.qwen.ai"
QWEN_DEVICE_CODE_URL = f"{QWEN_OAUTH_BASE}/api/v1/oauth2/device/code"
QWEN_TOKEN_URL = f"{QWEN_OAUTH_BASE}/api/v1/oauth2/token"
QWEN_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_SCOPE = "openid profile email model.completion"
QWEN_DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
QWEN_USER_AGENT = "qwen-code/1.0.0"
QWEN_DEFAULT_API_BASE = "https://portal.qwen.ai/v1"


class QwenPortalAuthError(BaseLLMException):
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


class GetAccessTokenError(QwenPortalAuthError):
    pass


class RefreshAccessTokenError(QwenPortalAuthError):
    pass
