"""MiniMax handler mixin for HeadroomProxy.

Routes MiniMax traffic (model names matching ``MiniMax-M*`` or
``minimax/MiniMax-M*``) to a dedicated upstream URL — independent of
the Anthropic upstream — so the two providers don't accidentally share
quota when only one is configured.

Wire format: MiniMax exposes an Anthropic-compatible /v1/messages API,
so the heavy lifting is delegated to AnthropicHandlerMixin. What this
mixin adds:

- **Upstream URL resolution**: prefers the explicit
  ``upstream_base_url`` arg, then ``ProxyConfig.minimax_api_url``, then
  the ``MINIMAX_TARGET_API_URL`` env var, then falls back to
  ``self.ANTHROPIC_API_URL``. The fallback is the legacy behaviour —
  pin a dedicated URL via ProxyConfig or env to route MiniMax traffic
  independently of Anthropic traffic.
- **Provider stamp**: forces ``provider_name="minimax"`` so the
  cost tracker and dashboard per-model breakdown attribute savings
  to MiniMax, not Anthropic.
- **Prefix strip**: removes the ``minimax/`` prefix from the model
  name before forwarding upstream, since the MiniMax gateway expects
  bare model names (``MiniMax-M3``, not ``minimax/MiniMax-M3``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse

from headroom.providers.minimax import MiniMaxProvider
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

logger = logging.getLogger("headroom.proxy.minimax")


class MiniMaxHandlerMixin:
    """Mixin providing MiniMax-specific proxy handler for HeadroomProxy.

    Routes traffic to the Anthropic handler (wire-compatible) but
    overrides the provider name, cost table, and upstream URL so
    M3/M2.7 traffic is bucketed correctly in the dashboard AND can
    be sent to a MiniMax-specific endpoint without disturbing the
    Anthropic upstream.
    """

    @staticmethod
    def _is_minimax_model(model: str) -> bool:
        """Return True if the given model name belongs to MiniMax.

        Accepts both bare (``MiniMax-M3``) and prefixed (``minimax/MiniMax-M3``)
        forms. Conservative: returns False for unknown models so
        Anthropic traffic is never accidentally routed here.
        """
        if not model:
            return False
        m = model.strip().lower()
        if m.startswith("minimax/"):
            return True
        return m.startswith("minimax-m") or "minimax-m" in m

    @staticmethod
    def _strip_minimax_prefix(model: str) -> str:
        """Strip the ``minimax/`` provider prefix from a model name."""
        if not model:
            return model
        if "/" in model:
            head, _, tail = model.partition("/")
            if head.lower() == "minimax":
                return tail
        return model

    def _resolve_minimax_upstream_url(
        self, override: str | None = None
    ) -> str:
        """Resolve the upstream base URL for MiniMax traffic.

        Priority (first match wins):
            1. ``override`` argument — caller-supplied URL.
            2. ``self.config.minimax_api_url`` — ProxyConfig field.
            3. ``MINIMAX_TARGET_API_URL`` env var.
            4. ``self.ANTHROPIC_API_URL`` — legacy fallback (used when
               the operator hasn't configured a separate MiniMax
               upstream; this is the historical default).

        Returns the resolved URL string, never empty.
        """
        if override:
            return override
        cfg_url = getattr(self.config, "minimax_api_url", None)
        if cfg_url:
            return cfg_url
        env_url = os.environ.get("MINIMAX_TARGET_API_URL")
        if env_url:
            return env_url
        # Late import: HeadroomProxy lives in headroom.proxy.server which
        # itself imports MiniMaxHandlerMixin, so a top-level import would
        # create a cycle.
        from headroom.proxy.server import HeadroomProxy

        return getattr(
            HeadroomProxy, "ANTHROPIC_API_URL", "https://api.anthropic.com"
        )

    async def handle_minimax_messages(
        self,
        request: Request,
        upstream_base_url: str | None = None,
        provider_name: str = "minimax",
        model_override: str | None = None,
        force_stream: bool = False,
    ) -> Response | StreamingResponse:
        """Handle ``POST /v1/messages`` for MiniMax traffic.

        Delegates to :meth:`AnthropicHandlerMixin.handle_anthropic_messages`
        because the wire format is identical, but:

        - Sets ``provider_name="minimax"`` so the request outcome is
          recorded with the correct provider and the dashboard
          per-model breakdown attributes savings to MiniMax, not
          Anthropic.
        - Strips the ``minimax/`` prefix from the model name before
          forwarding upstream, since the MiniMax gateway expects
          bare model names (``MiniMax-M3``, not ``minimax/MiniMax-M3``).
        - Routes to a MiniMax-specific upstream URL when one is
          configured (via ``ProxyConfig.minimax_api_url`` or the
          ``MINIMAX_TARGET_API_URL`` env var). Without this, MiniMax
          traffic would share Anthropic's upstream and any quota or
          rate-limit applied to that upstream would accidentally
          throttle MiniMax as well.

        Operators who want MiniMax to share the Anthropic upstream
        (e.g. to use a single Mavis Code gateway for both providers)
        can simply leave both env vars unset and the handler falls
        back to ``self.ANTHROPIC_API_URL``.
        """
        # Strip the prefix from the incoming model name so the upstream
        # gateway recognises it. Use a fresh request body if needed.
        if model_override is None:
            try:
                body_bytes = await request.body()
                parsed = json.loads(body_bytes or b"{}")
                if isinstance(parsed, dict) and "model" in parsed:
                    parsed["model"] = self._strip_minimax_prefix(
                        parsed["model"]
                    )
                    new_body = json.dumps(parsed).encode()
                    try:
                        request._body = new_body  # type: ignore[attr-defined]
                    except AttributeError:
                        # Fallback: rely on Anthropic handler parsing the
                        # body and patching model via model_override.
                        model_override = parsed["model"]
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "minimax: could not parse body for model strip"
                )

        # Resolve the MiniMax-specific upstream URL.
        resolved_url = self._resolve_minimax_upstream_url(upstream_base_url)

        # Delegate to Anthropic handler with the resolved MiniMax URL.
        return await AnthropicHandlerMixin.handle_anthropic_messages(
            self,
            request,
            upstream_base_url=resolved_url,
            provider_name=provider_name,  # "minimax" — overrides default
            model_override=model_override,
            force_stream=force_stream,
        )


__all__ = [
    "MiniMaxHandlerMixin",
    "MiniMaxProvider",
]
