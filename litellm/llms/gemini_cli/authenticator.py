"""
Gemini CLI OAuth authenticator.

Handles OAuth 2.0 token management for Gemini CLI, supporting:
  - Token loading from auth file
  - Token refresh via Google OAuth2
  - Browser-based OAuth login with PKCE (fallback)

Follows the same pattern as OpenClaw's google-gemini-cli-auth extension
which extracts OAuth client credentials from the Gemini CLI binary and
uses them to authenticate against Google's Code Assist API.
"""

import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    GEMINI_CLI_API_BASE,
    GEMINI_CLI_AUTH_URL,
    GEMINI_CLI_CODE_ASSIST_URL,
    GEMINI_CLI_OAUTH_SCOPES,
    GEMINI_CLI_TOKEN_URL,
    GEMINI_CLI_USERINFO_URL,
    GetAccessTokenError,
    RefreshAccessTokenError,
)

TOKEN_EXPIRY_BUFFER_SECONDS = 5 * 60
OAUTH_CALLBACK_PORT = 8085
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}/oauth2callback"
OAUTH_TIMEOUT_SECONDS = 5 * 60
OAUTH_CLIENT_ID_FIELD = "oauth_client_id"
OAUTH_CLIENT_SECRET_FIELD = "oauth_client_secret"

# Patterns to extract OAuth client credentials from the Gemini CLI binary
CLIENT_ID_PATTERN = re.compile(r"\d+-[a-z0-9]+\.apps\.googleusercontent\.com")
CLIENT_SECRET_PATTERN = re.compile(r"GOCSPX-[A-Za-z0-9_-]+")


class Authenticator:
    def __init__(self) -> None:
        self.auth_filename = os.getenv("GEMINI_CLI_AUTH_FILE", "auth.gemini_cli.json")
        self.token_dir = os.getenv(
            "GEMINI_CLI_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/gemini_cli"),
        )
        self.auth_file = os.path.join(self.token_dir, self.auth_filename)
        self._ensure_token_dir()
        self._client_id: Optional[str] = None
        self._client_secret: Optional[str] = None

    def get_api_base(self) -> str:
        return (
            os.getenv("GEMINI_CLI_API_BASE")
            or os.getenv("GOOGLE_GEMINI_CLI_API_BASE")
            or GEMINI_CLI_API_BASE
        )

    def get_access_token(self) -> str:
        auth_data = self._read_auth_file()
        if auth_data:
            access_token = auth_data.get("access_token")
            if access_token and not self._is_token_expired(auth_data):
                return access_token
            refresh_token = auth_data.get("refresh_token")
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token, auth_data=auth_data)
                    return refreshed["access_token"]
                except (RefreshAccessTokenError, GetAccessTokenError) as exc:
                    verbose_logger.warning(
                        "Gemini CLI refresh token failed, re-login required: %s", exc
                    )
                    raise GetAccessTokenError(
                        message=str(exc),
                        status_code=getattr(exc, "status_code", 401),
                    )

        raise GetAccessTokenError(
            message=(
                "Gemini CLI OAuth tokens are missing or expired. "
                "Run 'python auth.py gemini' from your LiteLLM config directory "
                "to authenticate, or refresh with 'python auth.py refresh gemini'."
            ),
            status_code=401,
        )

    def get_project_id(self) -> Optional[str]:
        auth_data = self._read_auth_file()
        if not auth_data:
            return None
        return auth_data.get("project_id")

    def _ensure_token_dir(self) -> None:
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _get_auth_file_candidates(self) -> list[str]:
        # Primary path used by LiteLLM provider config
        candidates = [self.auth_file]

        # Compatibility with litellmctl auth flow default output (~/.litellm/auth.gemini_cli.json)
        home_litellm_auth = os.path.join(
            os.path.expanduser("~/.litellm"), self.auth_filename
        )
        if home_litellm_auth not in candidates:
            candidates.append(home_litellm_auth)

        return candidates

    def _read_auth_file(self) -> Optional[Dict[str, Any]]:
        for candidate in self._get_auth_file_candidates():
            try:
                with open(candidate, "r") as f:
                    auth_data = json.load(f)
                # Pin to the discovered auth file path so refresh writes back there.
                self.auth_file = candidate
                self.token_dir = os.path.dirname(candidate)
                return auth_data
            except IOError:
                continue
            except json.JSONDecodeError as exc:
                verbose_logger.warning(
                    "Invalid Gemini CLI auth file %s: %s", candidate, exc
                )
                continue
        return None

    def _write_auth_file(self, data: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.auth_file), exist_ok=True)
            with open(self.auth_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as exc:
            verbose_logger.error("Failed to write Gemini CLI auth file: %s", exc)

    def _is_token_expired(self, auth_data: Dict[str, Any]) -> bool:
        expires_at = auth_data.get("expires_at")
        if expires_at is None:
            return True
        return time.time() >= float(expires_at) - TOKEN_EXPIRY_BUFFER_SECONDS

    # ── OAuth client credential resolution ──

    def _resolve_oauth_credentials(
        self, auth_data: Optional[Dict[str, Any]] = None
    ) -> tuple[str, str]:
        """
        Resolve OAuth client ID and secret.

        Strategy (in order):
        1. Environment variables
        2. Stored auth file metadata
        3. Extract from installed Gemini CLI binary
        """
        stored_auth_data = auth_data or self._read_auth_file() or {}
        stored_client_id = stored_auth_data.get(OAUTH_CLIENT_ID_FIELD)
        stored_client_secret = stored_auth_data.get(OAUTH_CLIENT_SECRET_FIELD)
        if stored_client_id and stored_client_secret:
            self._client_id = stored_client_id
            self._client_secret = stored_client_secret
            return stored_client_id, stored_client_secret

        if self._client_id and self._client_secret:
            return self._client_id, self._client_secret

        client_id = os.getenv("GEMINI_CLI_OAUTH_CLIENT_ID")
        client_secret = os.getenv("GEMINI_CLI_OAUTH_CLIENT_SECRET")

        if client_id and client_secret:
            self._client_id = client_id
            self._client_secret = client_secret
            return client_id, client_secret

        extracted = self._extract_credentials_from_cli()
        if extracted:
            self._client_id, self._client_secret = extracted
            return extracted

        raise GetAccessTokenError(
            message=(
                "Could not resolve Gemini CLI OAuth credentials. "
                "Either set GEMINI_CLI_OAUTH_CLIENT_ID and GEMINI_CLI_OAUTH_CLIENT_SECRET "
                "environment variables, or install the Gemini CLI: "
                "npm install -g @google/gemini-cli"
            ),
            status_code=401,
        )

    def _find_oauth2_js(self) -> Optional[str]:
        """Find gemini-cli-core oauth2.js across npm/bun/pnpm/yarn layouts."""
        oauth_subpath = os.path.join(
            "@google", "gemini-cli-core", "dist", "src", "code_assist", "oauth2.js"
        )

        gemini_bin = shutil.which("gemini")
        if gemini_bin:
            real_path = os.path.realpath(gemini_bin)
            current_dir = os.path.dirname(real_path)
            for _ in range(10):
                candidate = os.path.join(current_dir, "node_modules", oauth_subpath)
                if os.path.isfile(candidate):
                    return candidate
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    break
                current_dir = parent_dir

        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, ".bun", "install", "global", "node_modules", oauth_subpath),
            os.path.join("/usr/local/lib/node_modules", oauth_subpath),
            os.path.join("/usr/lib/node_modules", oauth_subpath),
            os.path.join(home, ".npm-global", "lib", "node_modules", oauth_subpath),
        ]

        nvm_dir = os.getenv("NVM_DIR", os.path.join(home, ".nvm"))
        nvm_versions_dir = os.path.join(nvm_dir, "versions", "node")
        if os.path.isdir(nvm_versions_dir):
            for version in os.listdir(nvm_versions_dir):
                candidates.append(
                    os.path.join(
                        nvm_versions_dir,
                        version,
                        "lib",
                        "node_modules",
                        oauth_subpath,
                    )
                )

        for path in candidates:
            if os.path.isfile(path):
                return path

        return None

    def _extract_credentials_from_cli(self) -> Optional[tuple]:
        """Extract OAuth client credentials from installed Gemini CLI sources."""
        oauth_file = self._find_oauth2_js()
        if not oauth_file:
            verbose_logger.debug("Gemini CLI oauth2.js not found")
            return None

        try:
            with open(oauth_file, "r") as f:
                content = f.read()

            client_id_match = CLIENT_ID_PATTERN.search(content)
            client_secret_match = CLIENT_SECRET_PATTERN.search(content)

            if client_id_match and client_secret_match:
                verbose_logger.debug(
                    "Extracted Gemini CLI OAuth credentials from %s",
                    oauth_file,
                )
                return client_id_match.group(), client_secret_match.group()
        except Exception as exc:
            verbose_logger.debug(
                "Failed to extract credentials from Gemini CLI oauth2.js: %s", exc
            )

        return None

    # ── PKCE OAuth login flow ──

    def _login_oauth_pkce(self) -> Dict[str, str]:
        """
        Perform OAuth 2.0 login with PKCE via browser.
        """
        client_id, client_secret = self._resolve_oauth_credentials()

        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            )
            .rstrip(b"=")
            .decode()
        )

        state = secrets.token_urlsafe(16)

        auth_params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": OAUTH_REDIRECT_URI,
            "scope": GEMINI_CLI_OAUTH_SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }

        auth_url = f"{GEMINI_CLI_AUTH_URL}?{urlencode(auth_params)}"

        print(  # noqa: T201
            "Sign in with Google for Gemini CLI:\n"
            f"1) Opening browser to: {auth_url}\n"
            "2) Authorize the application\n"
            "3) Waiting for callback...",
            flush=True,
        )

        authorization_code = self._capture_oauth_callback(auth_url, state)

        tokens = self._exchange_code(
            authorization_code, code_verifier, client_id, client_secret
        )

        project_id = self._discover_project(tokens["access_token"])

        auth_data = self._build_auth_record(tokens, project_id)
        self._write_auth_file(auth_data)

        return tokens

    def _capture_oauth_callback(self, auth_url: str, expected_state: str) -> str:
        """Start local HTTP server and capture OAuth callback."""
        result = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                code = params.get("code", [None])[0]
                state = params.get("state", [None])[0]
                error = params.get("error", [None])[0]

                if error:
                    result["error"] = error
                elif state != expected_state:
                    result["error"] = "State mismatch"
                elif code:
                    result["code"] = code

                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication successful!</h2>"
                    b"<p>You can close this tab and return to your terminal.</p>"
                    b"</body></html>"
                )

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("localhost", OAUTH_CALLBACK_PORT), CallbackHandler)
        server.timeout = OAUTH_TIMEOUT_SECONDS

        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()

        if "error" in result:
            raise GetAccessTokenError(
                message=f"OAuth callback error: {result['error']}",
                status_code=401,
            )

        if "code" not in result:
            raise GetAccessTokenError(
                message="OAuth callback timed out. No authorization code received.",
                status_code=408,
            )

        return result["code"]

    def _exchange_code(
        self,
        code: str,
        code_verifier: str,
        client_id: str,
        client_secret: str,
    ) -> Dict[str, str]:
        """Exchange authorization code for tokens."""
        try:
            client = _get_httpx_client()
            resp = client.post(
                GEMINI_CLI_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH_REDIRECT_URI,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code_verifier": code_verifier,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: {exc}",
                status_code=400,
            )

        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 3600)

        if not access_token:
            raise GetAccessTokenError(
                message=f"Token exchange response missing access_token: {data}",
                status_code=400,
            )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token or "",
            "expires_in": expires_in,
        }

    def _refresh_tokens(
        self, refresh_token: str, auth_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Refresh an expired access token."""
        client_id, client_secret = self._resolve_oauth_credentials(auth_data=auth_data)

        try:
            client = _get_httpx_client()
            resp = client.post(
                GEMINI_CLI_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: {exc}",
                status_code=400,
            )

        access_token = data.get("access_token")
        if not access_token:
            raise RefreshAccessTokenError(
                message=f"Refresh response missing access_token: {data}",
                status_code=400,
            )

        expires_in = data.get("expires_in", 3600)
        new_refresh = data.get("refresh_token", refresh_token)

        auth_record = {
            **(auth_data or self._read_auth_file() or {}),
            OAUTH_CLIENT_ID_FIELD: client_id,
            OAUTH_CLIENT_SECRET_FIELD: client_secret,
            "access_token": access_token,
            "refresh_token": new_refresh,
            "expires_at": time.time() + int(expires_in) - TOKEN_EXPIRY_BUFFER_SECONDS,
        }
        self._write_auth_file(auth_record)

        return {"access_token": access_token, "refresh_token": new_refresh}

    def _discover_project(self, access_token: str) -> Optional[str]:
        """
        Discover the user's Google Cloud project via Code Assist API.

        Strategy:
        1. Check GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_PROJECT_ID env vars
        2. Call loadCodeAssist (with project hint if available)
        3. Fall back to listing GCP projects via Resource Manager API
        """
        env_project = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        )

        try:
            client = _get_httpx_client()
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            # Call loadCodeAssist with project hint
            body = {}
            if env_project:
                body["cloudaicompanionProject"] = env_project
            resp = client.post(
                f"{GEMINI_CLI_CODE_ASSIST_URL}/v1internal:loadCodeAssist",
                headers=headers,
                json=body,
            )
            if resp.status_code == 200:
                data = resp.json()
                pid = (
                    data.get("cloudaicompanionProject")
                    or data.get("projectId")
                    or data.get("project_id")
                )
                if pid:
                    verbose_logger.debug("Discovered project: %s", pid)
                    return pid if isinstance(pid, str) else pid.get("id")

            # Fall back to listing GCP projects
            if not env_project:
                resp2 = client.get(
                    "https://cloudresourcemanager.googleapis.com/v1/projects",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if resp2.status_code == 200:
                    projects = resp2.json().get("projects", [])
                    for p in projects:
                        if "gemini" in p.get("name", "").lower() or "lang" in p.get("projectId", ""):
                            verbose_logger.debug("Using GCP project: %s", p["projectId"])
                            return p["projectId"]
                    if projects:
                        verbose_logger.debug("Using first GCP project: %s", projects[0]["projectId"])
                        return projects[0]["projectId"]

        except Exception as exc:
            verbose_logger.debug("Project discovery failed (non-fatal): %s", exc)

        return env_project

    def _build_auth_record(
        self, tokens: Dict[str, Any], project_id: Optional[str]
    ) -> Dict[str, Any]:
        expires_in = tokens.get("expires_in", 3600)
        auth_record = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token", ""),
            "expires_at": time.time() + int(expires_in) - TOKEN_EXPIRY_BUFFER_SECONDS,
            "project_id": project_id,
        }
        client_id = self._client_id or os.getenv("GEMINI_CLI_OAUTH_CLIENT_ID")
        client_secret = self._client_secret or os.getenv("GEMINI_CLI_OAUTH_CLIENT_SECRET")
        if client_id and client_secret:
            auth_record[OAUTH_CLIENT_ID_FIELD] = client_id
            auth_record[OAUTH_CLIENT_SECRET_FIELD] = client_secret
        return auth_record
