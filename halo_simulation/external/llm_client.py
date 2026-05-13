"""Thin Anthropic / OpenAI-compatible client for HALO LLM specialist cycles (direct Anthropic or LiteLLM)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from halo_simulation import config

logger = logging.getLogger(__name__)


def _anthropic_error_message(response: httpx.Response | None) -> str | None:
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        m = err.get("message")
        return str(m).strip() if m else None
    if isinstance(err, str) and err.strip():
        return err.strip()
    return None


def _openai_style_error_message(response: httpx.Response | None) -> str | None:
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        m = err.get("message")
        return str(m).strip() if m else None
    return None


def _http_error_detail(response: httpx.Response | None) -> str | None:
    return _anthropic_error_message(response) or _openai_style_error_message(response)


def _parse_json_dict_from_llm_text(raw_text: str) -> dict[str, Any]:
    """
    Parse a single JSON object from model output.

    LiteLLM / Bedrock often prefix prose or wrap fences despite "JSON only" prompts; ``json.loads``
    on the whole string then fails with errors like "Expecting ',' delimiter".
    """
    s = raw_text.strip()
    s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    i = s.find("{")
    if i < 0:
        raise ValueError("No JSON object start '{' in LLM output")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(s[i:])
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root must be a JSON object")
    return obj


def _extract_text_from_llm_response(body: dict[str, Any], protocol: str) -> str:
    if protocol == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-style response missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("OpenAI-style response invalid choices[0]")
        msg = first.get("message")
        if not isinstance(msg, dict):
            raise ValueError("OpenAI-style response missing message")
        c = msg.get("content")
        if isinstance(c, list):
            chunks: list[str] = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    chunks.append(part)
            return "".join(chunks).strip()
        return str(c or "").strip()

    block = body.get("content")
    if not isinstance(block, list) or not block:
        raise ValueError("Anthropic response missing content")
    first_b = block[0]
    if not isinstance(first_b, dict):
        raise ValueError("Anthropic response invalid content[0]")
    return str(first_b.get("text") or "").strip()


def _format_llm_failure(
    exc: Exception,
    response: httpx.Response | None = None,
    request: httpx.Request | None = None,
) -> str:
    """Short UI-safe explanation for the reasoning feed (no secrets)."""
    req_host = ""
    try:
        if request is not None and request.url:
            req_host = str(request.url.host or "")
        elif isinstance(exc, httpx.HTTPStatusError) and exc.request and exc.request.url:
            req_host = str(exc.request.url.host or "")
    except Exception:
        req_host = ""
    hit_anthropic_direct = req_host == "api.anthropic.com"
    auth_style = os.getenv("LLM_AUTH_STYLE", "x-api-key").strip().lower()
    proto = config.llm_protocol()

    if isinstance(exc, httpx.HTTPStatusError):
        resp = exc.response
        code = resp.status_code
        api_msg = _http_error_detail(resp)
        if code == 401:
            line = "401 Unauthorized — API key rejected."
            if api_msg:
                line += f" ({api_msg})"
            if proto == "openai":
                line += (
                    " LiteLLM OpenAI mode: confirm LITELLM_BASE_URL, Bearer token in CLAUDE_KEY, "
                    "and LLM_MODEL (e.g. anthropic/claude-3-5-haiku-20241022)."
                )
            elif hit_anthropic_direct:
                line += (
                    " Traffic is going to api.anthropic.com (LiteLLM base URL not applied — "
                    "add LLM_ANTHROPIC_BASE_URL or LITELLM_BASE_URL to .env and restart uvicorn). "
                    "If you stay on Anthropic directly, use a valid sk-ant-api03… key from "
                    "console.anthropic.com. If you use Cisco LiteLLM: put proxy URL in "
                    "LLM_ANTHROPIC_BASE_URL, virtual key in CLAUDE_KEY, often LLM_AUTH_STYLE=bearer "
                    "— or use LLM_PROTOCOL=openai for OpenAI-compatible LiteLLM."
                )
            else:
                line += (
                    f" Host was {req_host or '?'}. Confirm LLM_ANTHROPIC_BASE_URL, "
                    "virtual key in CLAUDE_KEY/ANTHROPIC_API_KEY, LLM_AUTH_STYLE=bearer if needed, restart server."
                )
                if auth_style not in ("bearer", "authorization", "litellm"):
                    line += f" (current LLM_AUTH_STYLE={auth_style!r})"
            return line[:480]
        if code == 403:
            line = "403 Forbidden — key may lack billing or model access."
            if api_msg:
                line += f" ({api_msg})"
            return line[:400]
        if api_msg:
            return f"HTTP {code}: {api_msg}"[:400]
        return f"HTTP {code}: {exc.request.method} failed"[:400]
    detail = str(exc).strip() or type(exc).__name__
    if len(detail) > 220:
        detail = detail[:217] + "…"
    return f"{type(exc).__name__}: {detail}"


class LLMClient:
    """
    LLM calls for the specialist: Anthropic ``/v1/messages`` or OpenAI-compatible ``/v1/chat/completions``
    (configure with ``LLM_PROTOCOL`` — LiteLLM commonly uses the latter).
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        messages_url: str | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model if model is not None else config.llm_model()
        self._protocol = config.llm_protocol()
        if messages_url is not None:
            self._post_url = messages_url.rstrip("/")
        elif self._protocol == "openai":
            self._post_url = config.llm_openai_chat_url().rstrip("/")
            if not self._post_url:
                logger.warning(
                    "LLM_PROTOCOL=openai but LITELLM_BASE_URL / LLM_ANTHROPIC_BASE_URL / LLM_OPENAI_CHAT_URL is empty",
                )
        else:
            self._post_url = config.llm_anthropic_messages_url().rstrip("/")

        try:
            from urllib.parse import urlparse

            host = urlparse(self._post_url).hostname or "?"
            auth = os.getenv("LLM_AUTH_STYLE", "x-api-key").strip() or "x-api-key"
            if self._protocol == "openai":
                logger.warning(
                    "LLMClient: protocol=openai POST host=%s model=%s (Bearer)",
                    host,
                    self._model,
                )
            elif host == "api.anthropic.com":
                logger.warning(
                    "LLMClient: will POST to api.anthropic.com (no LiteLLM base in env). model=%s auth=%s",
                    self._model,
                    auth,
                )
            else:
                logger.warning(
                    "LLMClient: Anthropic Messages API host=%s model=%s LLM_AUTH_STYLE=%s",
                    host,
                    self._model,
                    auth,
                )
        except Exception:
            pass

    def reason(self, prompt: str) -> dict[str, Any]:
        """
        Sends prompt to Claude, expects JSON response.
        Returns parsed dict: {"relevant": bool, "api_id": str, "reason": str}
        On missing key or request/parse errors, returns a stable dict with a human-readable reason
        (shown in the dashboard "Agent reasoning" feed).
        """
        if not (self._api_key or "").strip():
            return {
                "relevant": False,
                "api_id": "none",
                "reason": "No Anthropic API key — set ANTHROPIC_API_KEY or CLAUDE_KEY (e.g. in repo .env) and restart",
            }
        if not self._post_url:
            return {
                "relevant": False,
                "api_id": "none",
                "reason": "LLM URL not configured — for LiteLLM set LITELLM_BASE_URL (and LLM_PROTOCOL=openai if using /v1/chat/completions)",
            }
        try:
            if self._protocol == "openai":
                payload: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                payload = {
                    "model": self._model,
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                }
            response = httpx.post(
                self._post_url,
                headers=config.llm_request_headers(self._api_key, protocol=self._protocol),
                json=payload,
                timeout=8.0,
            )
            response.raise_for_status()
            raw_text = _extract_text_from_llm_response(response.json(), self._protocol)
            return _parse_json_dict_from_llm_text(raw_text)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 401:
                logger.debug("LLM reasoning HTTP 401 (shown in UI): %s", e)
            else:
                logger.warning("LLM reasoning HTTP error: %s", e)
            return {
                "relevant": False,
                "api_id": "none",
                "reason": _format_llm_failure(e, e.response, e.request),
            }
        except json.JSONDecodeError as e:
            logger.warning("LLM reasoning JSON parse failed: %s", e)
            return {
                "relevant": False,
                "api_id": "none",
                "reason": f"LLM returned invalid JSON ({e})",
            }
        except Exception as e:
            logger.warning("LLM reasoning failed: %s", e)
            return {
                "relevant": False,
                "api_id": "none",
                "reason": _format_llm_failure(e),
            }

    def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 400,
        timeout: float = 10.0,
    ) -> dict[str, Any] | None:
        """
        General JSON completion (interpretation, structured extraction).
        Returns parsed dict or None on error / missing API key.
        """
        if not (self._api_key or "").strip():
            return None
        if not self._post_url:
            return None
        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = httpx.post(
                self._post_url,
                headers=config.llm_request_headers(self._api_key, protocol=self._protocol),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            raw_text = _extract_text_from_llm_response(response.json(), self._protocol)
            return _parse_json_dict_from_llm_text(raw_text)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 401:
                logger.debug("LLM complete_json HTTP 401: %s", e)
            else:
                logger.warning("LLM complete_json HTTP error: %s", e)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("LLM complete_json parse failed: %s", e)
            return None
        except Exception as e:
            logger.warning("LLM complete_json failed: %s", e)
            return None
