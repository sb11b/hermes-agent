"""xAI-Live voice chat mode: the full-duplex voice frontend that delegates to Hermes.

``voice.voice_chat_mode: xai-live`` replaces the chained STT → turn → TTS loop with ONE
full-duplex voice model (xAI ``grok-voice-think-fast-2.0``) that owns the microphone and the
speaker and delegates every real request to Hermes as its brain. Hermes stays the agent:
whatever model/provider the session has selected answers, with the full toolset. This
parallels the OpenAI ``gpt-live`` (WebRTC) mode in :mod:`tools.voice_live`; the delegation
contract (``VOICE_LIVE_TURN_NOTE``) is shared, only the transport differs.

Division of labour (the xAI realtime model has no tools of its own):

* the desktop renderer holds the WebSocket media session (PCM16 24 kHz in and out) and the
  session.update payload;
* this module resolves WHICH model/voice to use and performs the one server-side step the
  API requires — minting a short-lived *ephemeral client secret* with the xAI key
  (``POST /v1/realtime/client_secrets``), so the key never reaches the client;
* the renderer opens ``wss://api.x.ai/v1/realtime`` with the subprotocol
  ``xai-client-secret.<token>`` and turns each delegation into a normal ``prompt.submit`` on
  the active session (surface ``voice-live``) — Hermes' answer is what the voice speaks.

Vendor contract: https://docs.x.ai/docs/guides/realtime (ephemeral client secrets +
WebSocket transport). Billing runs on the xAI key, separate from the Hermes turn.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

# The delegation contract is shared with the gpt-live path: import, never duplicate.
from tools.voice_live import (
    VOICE_LIVE_TURN_NOTE,
    _voice_section,
    voice_live_turn_note,
)

logger = logging.getLogger(__name__)

XAI_LIVE_MODE = "xai-live"

# Spelling variants accepted in config, mirroring the gpt-live set so its parsing is unchanged.
XAI_LIVE_MODE_ALIASES = ("xai", "grok-voice")
GPT_LIVE_MODE = "gpt-live"
GPT_LIVE_MODE_ALIASES = ("gptlive", "live")
CHAINED_MODE = "chained"

# Pinned to the versioned flagship ("Grok Voice Think Fast 2.0"); the reference client pins
# it too so an alias move on the vendor side can't shift the voice stack.
DEFAULT_XAI_LIVE_MODEL = "grok-voice-think-fast-2.0"
DEFAULT_XAI_LIVE_VOICE = ""  # empty = the xAI default voice
DEFAULT_XAI_LIVE_BASE_URL = "https://api.x.ai/v1"
DEFAULT_EPHEMERAL_TOKEN_TTL_SECONDS = 300

XAI_CLIENT_SECRETS_URL = f"{DEFAULT_XAI_LIVE_BASE_URL}/realtime/client_secrets"
XAI_REALTIME_WS_URL = f"wss://api.x.ai/v1/realtime?model={DEFAULT_XAI_LIVE_MODEL}"

XAI_TRANSPORT_TYPE = "xai-websocket"
# WebSocket subprotocol prefix carrying the ephemeral client secret to the vendor.
XAI_CLIENT_SECRET_SUBPROTOCOL_PREFIX = "xai-client-secret."


def resolve_live_mode(voice: Optional[dict[str, Any]] = None) -> str:
    """``chained`` (default), ``gpt-live`` or ``xai-live``. Underscore spellings normalise to
    hyphens; ``xai``/``grok-voice`` are accepted aliases for the xAI mode. The gpt-live
    variant set is the same one :func:`tools.voice_live.voice_chat_mode` accepts, and an
    unrecognised value falls back to ``chained`` exactly as it does there."""
    raw = (voice if voice is not None else _voice_section()).get("voice_chat_mode")
    mode = str(raw or CHAINED_MODE).strip().lower().replace("_", "-")
    if mode in {XAI_LIVE_MODE, *XAI_LIVE_MODE_ALIASES}:
        return XAI_LIVE_MODE
    if mode in {GPT_LIVE_MODE, *GPT_LIVE_MODE_ALIASES}:
        return GPT_LIVE_MODE
    return CHAINED_MODE


def _xai_live_section(voice: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    section = (voice if voice is not None else _voice_section()).get("xai_live")
    return section if isinstance(section, dict) else {}


def _resolve_api_key(live: dict[str, Any]) -> str:
    """The xAI API key: ``voice.xai_live.api_key`` first, else ``XAI_API_KEY`` resolved
    through the runtime (config → profile scope → env/.env → pool) so scoped deployments
    behave like every other provider. OAuth credentials are NOT used here — this endpoint
    mints API-metered client secrets."""
    from tools.xai_http import _resolve_explicit_xai_api_key
    return str(live.get("api_key") or "").strip() or _resolve_explicit_xai_api_key()


def resolve_xai_live_status() -> dict[str, Any]:
    """Non-secret readiness verdict for the client: which mode is selected and whether the
    xAI live layer can start (a key resolves). Never returns the key or a token."""
    voice = _voice_section()
    live = _xai_live_section(voice)
    api_key = _resolve_api_key(live)
    return {
        "mode": resolve_live_mode(voice),
        "available": bool(api_key),
        "reason": (
            None
            if api_key
            else "no xAI API key (set XAI_API_KEY or voice.xai_live.api_key)"
        ),
        "model": str(live.get("model") or DEFAULT_XAI_LIVE_MODEL),
        "voice": str(live.get("voice") or DEFAULT_XAI_LIVE_VOICE),
    }


def xai_live_model() -> str:
    """The configured (or default) xAI realtime model string."""
    return str(_xai_live_section().get("model") or DEFAULT_XAI_LIVE_MODEL)


def xai_realtime_ws_url(model: Optional[str] = None) -> str:
    """The realtime WebSocket URL the renderer connects to (no credentials in the query)."""
    return f"wss://api.x.ai/v1/realtime?model={model or xai_live_model()}"


def mint_ephemeral_token() -> str:
    """Mint a short-lived xAI realtime client secret and return its ``client_secret`` value.

    ``POST /v1/realtime/client_secrets`` with ``Authorization: Bearer <key>`` and
    ``{"expires_after": {"seconds": 300}}``. Uses ``httpx`` when available, else ``urllib``.
    Raises ``ValueError`` when no key resolves and ``RuntimeError`` (carrying the vendor
    status and detail) when the request is rejected. The key and the returned token are
    never logged.
    """
    api_key = _resolve_api_key(_xai_live_section())
    if not api_key:
        raise ValueError(
            "xAI-Live needs an xAI API key (XAI_API_KEY or voice.xai_live.api_key)"
        )

    payload = json.dumps(
        {"expires_after": {"seconds": DEFAULT_EPHEMERAL_TOKEN_TTL_SECONDS}}
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        import httpx  # type: ignore
    except ImportError:
        data, status_code, detail = _mint_with_urllib(XAI_CLIENT_SECRETS_URL, payload, headers)
    else:
        data, status_code, detail = _mint_with_httpx(httpx, XAI_CLIENT_SECRETS_URL, payload, headers)

    if status_code is not None:
        # Detail only — never the request headers (the bearer) or the response body echo.
        raise RuntimeError(f"xAI client secret request failed ({status_code}): {detail}")

    if not isinstance(data, dict):
        raise RuntimeError(
            "xAI client secret request failed (502): unexpected response from api.x.ai"
        )

    # Top-level fields first, then the nested client_secret object.
    token = data.get("value") or data.get("client_secret")
    if isinstance(token, dict):
        token = token.get("value")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(
            "xAI client secret request failed (502): no client_secret value in response"
        )
    token = token.strip()
    # Handed to the renderer as the WebSocket subprotocol only; never logged.
    return token


def _mint_with_httpx(httpx_mod: Any, url: str, payload: bytes, headers: Dict[str, str]):
    """Mirror of the reference client (``voice_realtime.py``): timeout/transport faults
    become 504/502 and an error status carries the vendor detail."""
    try:
        with httpx_mod.Client(timeout=httpx_mod.Timeout(30.0, connect=10.0)) as client:
            response = client.post(url, headers=headers, content=payload)
    except httpx_mod.TimeoutException as exc:
        raise RuntimeError(
            "xAI client secret request failed (504): xAI realtime secrets service timed out."
        ) from exc
    except httpx_mod.HTTPError as exc:
        raise RuntimeError(
            "xAI client secret request failed (502): could not reach the xAI realtime "
            "secrets service."
        ) from exc

    if response.status_code >= 400:
        return None, response.status_code, _xai_error_detail(response.text)
    try:
        return response.json(), None, ""
    except ValueError:
        return None, 502, "xAI realtime secrets returned unreadable data."


def _mint_with_urllib(url: str, payload: bytes, headers: Dict[str, str]):
    """Same contract as the httpx path, on the stdlib client."""
    request = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", "replace")
        logger.warning("xAI client secret request failed: HTTP %s", exc.code)
        return None, exc.code, _xai_error_detail(err_body)
    except urllib.error.URLError as exc:
        logger.warning("xAI client secret request failed: %s", type(exc).__name__)
        return None, 502, "could not reach the xAI realtime secrets service."
    # Mirror the httpx path: a non-JSON 200 is a 502 "unreadable data", never a
    # bare JSONDecodeError (which the route's `except ValueError` would mis-map
    # to a 503 with a vendor parser message).
    try:
        return json.loads(body), None, ""
    except ValueError:
        return None, 502, "xAI realtime secrets returned unreadable data."


def _xai_error_detail(body: str) -> str:
    """Best-effort vendor message from an error body (mirrors ``_raise_for_realtime_error``).
    Never includes the request headers."""
    detail = "xAI realtime secrets request failed."
    try:
        parsed = json.loads(body)
    except ValueError:
        return detail
    if not isinstance(parsed, dict):
        return detail
    err = parsed.get("error") or parsed.get("detail") or parsed.get("message")
    if isinstance(err, dict):
        err = err.get("message") or str(err)
    if isinstance(err, str) and err.strip():
        detail = err.strip()[:240]
    return detail


def build_xai_session_handshake(history: Optional[list] = None) -> dict[str, Any]:
    """Hand the renderer everything it needs to open the xAI realtime WebSocket.

    Returns ``{"transport": {"type": "xai-websocket", "token": <ephemeral client secret>,
    "url": "wss://api.x.ai/v1/realtime?model=…"}, "model": …}``. The renderer uses the token
    as the WebSocket subprotocol ``xai-client-secret.<token>``; the xAI key itself never
    reaches the client, the same discipline as the gpt-live SDP path. ``history`` is not
    sent to the vendor (the live layer seeds context from the conversation itself, and the
    Hermes turn owns the real transcript) — it stays on the signature for parity with
    :func:`tools.voice_live.create_webrtc_session`.

    Raises ``ValueError`` (no key) or ``RuntimeError`` (vendor rejection), same as the
    gpt-live path, so the route maps them identically.
    """
    model = xai_live_model()
    token = mint_ephemeral_token()
    handshake: Dict[str, Any] = {
        "transport": {
            "type": XAI_TRANSPORT_TYPE,
            "token": token,
            "url": xai_realtime_ws_url(model),
            "subprotocol": f"{XAI_CLIENT_SECRET_SUBPROTOCOL_PREFIX}{token}",
        },
        "model": model,
    }
    return handshake


__all__ = [
    "XAI_LIVE_MODE",
    "XAI_TRANSPORT_TYPE",
    "VOICE_LIVE_TURN_NOTE",
    "voice_live_turn_note",
    "resolve_live_mode",
    "resolve_xai_live_status",
    "xai_live_model",
    "xai_realtime_ws_url",
    "mint_ephemeral_token",
    "build_xai_session_handshake",
]
