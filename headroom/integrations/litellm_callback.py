"""LiteLLM callback — add Headroom compression to LiteLLM with one line.

    import litellm
    from headroom.integrations.litellm_callback import HeadroomCallback

    litellm.callbacks = [HeadroomCallback()]
    # All LiteLLM calls now get compressed automatically.

    # Or with custom config:
    litellm.callbacks = [HeadroomCallback(min_tokens=1000)]

Works with LiteLLM's completion(), acompletion(), and proxy modes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HeadroomCallback:
    """LiteLLM callback that compresses messages before each API call.

    Implements LiteLLM's CustomLogger interface (async_pre_call_hook).
    Compresses messages using Headroom's full pipeline, reducing token
    usage across all providers LiteLLM supports.

    Usage:
        import litellm
        from headroom.integrations.litellm_callback import HeadroomCallback

        litellm.callbacks = [HeadroomCallback()]
        response = litellm.completion(model="gpt-4o", messages=[...])
    """

    def __init__(
        self,
        min_tokens: int = 500,
        model_limit: int = 200000,
        hooks: Any = None,
    ) -> None:
        self._min_tokens = min_tokens
        self._model_limit = model_limit
        self._hooks = hooks
        self._total_saved = 0

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls."""
        return self._total_saved

    async def async_pre_call_hook(
        self,
        user_api_key: str,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Called by LiteLLM before each API call. Compresses messages."""
        if call_type not in ("completion", "acompletion"):
            return data

        messages = data.get("messages", [])
        model = data.get("model", "")

        if not messages:
            return data

        try:
            from headroom.compress import compress

            result = compress(
                messages=messages,
                model=model or "claude-sonnet-4-5-20250929",
                model_limit=self._model_limit,
                hooks=self._hooks,
            )

            if result.tokens_saved > 0:
                data["messages"] = result.messages
                self._total_saved += result.tokens_saved
                logger.info(
                    "Headroom: %d→%d tokens (saved %d, %.0f%%) [total saved: %d]",
                    result.tokens_before,
                    result.tokens_after,
                    result.tokens_saved,
                    result.compression_ratio * 100,
                    self._total_saved,
                )

        except Exception as e:
            logger.warning("Headroom compression failed, using original messages: %s", e)

        return data

    async def async_success_handler(
        self, kwargs: dict, response: Any, start_time: Any, end_time: Any
    ) -> None:
        """Called after successful completion. No-op for now."""
        pass

    async def async_failure_handler(
        self, kwargs: dict, response: Any, start_time: Any, end_time: Any
    ) -> None:
        """Called after failed completion. No-op for now."""
        pass
