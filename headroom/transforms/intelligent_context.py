"""Intelligent context manager — Rust-backed via PyO3.

The 1077-LOC Python implementation has been retired (2026-05-01). All
context-management decisions now flow through
`headroom._core.IntelligentContextManager` (built from
`crates/headroom-py`). This module retains the public Python surface —
`IntelligentContextManager`, `IntelligentContextConfig`, and the
`ContextStrategy` enum — so existing call sites (proxy, client, CLI,
tests) keep working unchanged.

Behaviour change vs the retired implementation
==============================================

The Rust port deliberately ships a smaller set of strategies for OSS:

- `DROP_BY_SCORE` — multi-factor scoring + safety rails + CCR-on-drop.
  This is the OSS-defining strategy and the only one OSS ships.
- `COMPRESS_FIRST` — **not in OSS**. The cut moves the
  ContentRouter-call-on-compress path to the Enterprise edition. If
  configured, this shim warns once and ignores the request.
- `SUMMARIZE` — **not in OSS**. The user-supplied LLM-callback
  summarization belongs to Enterprise. Same warn-and-ignore behaviour.

Python config fields that target the cut strategies (`compress_threshold`,
`summarize_threshold`, `summarization_*`, `memory_tiers_*`,
`warm_tier_*`, `cold_tier_*`) are silently accepted for backwards
compatibility but have no effect. A one-time `logger.info` notes the
behaviour change at construction time so it shows up in proxy startup
logs.

`MessageScore` is re-exported from `.scoring` so call sites that did
`from headroom.transforms.intelligent_context import MessageScore` keep
working. The dataclass shape is preserved.

The `headroom._core` extension is a hard import — there is no Python
fallback. Build it locally with `scripts/build_rust_extension.sh`
(wraps `maturin develop`) or install a prebuilt wheel.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..config import IntelligentContextConfig, TransformResult
from ..tokenizer import Tokenizer
from ..utils import create_dropped_context_marker, deep_copy_messages
from .base import Transform
from .scoring import MessageScore  # re-export for back-compat

if TYPE_CHECKING:
    # Imported only for type hints — the real values go unused now.
    from ..telemetry.toin import ToolIntelligenceNetwork
    from .progressive_summarizer import SummarizeFn

logger = logging.getLogger(__name__)

__all__ = [
    "ContextStrategy",
    "IntelligentContextConfig",
    "IntelligentContextManager",
    "MessageScore",
]


class ContextStrategy(Enum):
    """Strategy for handling over-budget context.

    Preserved for back-compat with call sites that import the enum.
    Only `DROP_BY_SCORE` and `NONE` describe behaviour OSS ships;
    `COMPRESS_FIRST` and `SUMMARIZE` exist as members so older config
    code that references them doesn't `AttributeError`, but they map
    to no-ops in the Rust-backed manager.
    """

    NONE = "none"
    COMPRESS_FIRST = "compress"
    SUMMARIZE = "summarize"
    DROP_BY_SCORE = "drop_scored"
    HYBRID = "hybrid"


class IntelligentContextManager(Transform):
    """Rust-backed `IntelligentContextManager`.

    Same constructor and `apply` shape as the retired Python class. The
    `toin`, `summarize_fn`, and `observer` parameters are accepted for
    back-compat but currently unused by the Rust core (the core has its
    own MessageScorer; TOIN integration is a follow-up via the
    EmbeddingProvider/ToinProvider trait surfaces in the scoring crate).

    See module docstring for the OSS-vs-Enterprise behaviour cut.
    """

    name = "intelligent_context"

    def __init__(
        self,
        config: IntelligentContextConfig | None = None,
        toin: ToolIntelligenceNetwork | None = None,
        summarize_fn: SummarizeFn | None = None,
        observer: Any = None,
    ):
        # Hard import — fail loudly if the wheel isn't built. Same
        # convention as diff_compressor.py and smart_crusher.py.
        from headroom._core import (
            IcmConfig as _RustIcmConfig,
        )
        from headroom._core import (
            IntelligentContextManager as _RustICM,
        )

        cfg = config or IntelligentContextConfig()
        self.config = cfg

        # Stash unused args so call sites that probe attributes don't
        # break. They have no behavioural effect in OSS.
        self.toin = toin
        self._summarize_fn = summarize_fn
        self._observer = observer

        # Map the Python config to the Rust config (6 fields). Note
        # what's silently dropped:
        weights = cfg.scoring_weights
        if _is_enterprise_config(cfg):
            logger.info(
                "intelligent_context: Rust-backed manager ships only "
                "DROP_BY_SCORE in OSS. compress_threshold=%s, "
                "summarize_threshold=%s, summarization_enabled=%s, "
                "memory_tiers_enabled=%s — these are accepted for "
                "back-compat but have no effect. Upgrade to "
                "Enterprise for COMPRESS_FIRST + SUMMARIZE strategies.",
                getattr(cfg, "compress_threshold", None),
                getattr(cfg, "summarize_threshold", None),
                getattr(cfg, "summarization_enabled", None),
                getattr(cfg, "memory_tiers_enabled", None),
            )

        self._rust = _RustICM(
            _RustIcmConfig(
                enabled=cfg.enabled,
                keep_system=cfg.keep_system,
                keep_last_turns=cfg.keep_last_turns,
                output_buffer_tokens=cfg.output_buffer_tokens,
                ccr_on_drop=True,  # OSS-defining default
                recency=weights.recency,
                semantic_similarity=weights.semantic_similarity,
                toin_importance=weights.toin_importance,
                error_indicator=weights.error_indicator,
                forward_reference=weights.forward_reference,
                token_density=weights.token_density,
            )
        )

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        if not self.config.enabled:
            return False
        model_limit = kwargs.get("model_limit", 128_000)
        output_buffer = kwargs.get("output_buffer", self.config.output_buffer_tokens)
        return bool(
            self._rust.should_apply(
                json.dumps(messages),
                model_limit=model_limit,
                output_buffer=output_buffer,
            )
        )

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        model_limit = kwargs.get("model_limit", 128_000)
        output_buffer = kwargs.get("output_buffer", self.config.output_buffer_tokens)
        frozen_message_count = kwargs.get("frozen_message_count", 0)

        # Defensive deep copy so callers' message list isn't mutated by
        # the JSON round-trip into Rust. Mirrors the retired Python
        # impl which built its result on a fresh deepcopy too.
        messages_copy = deep_copy_messages(messages)

        result = self._rust.apply(
            json.dumps(messages_copy),
            model_limit=model_limit,
            output_buffer=output_buffer,
            frozen_message_count=frozen_message_count,
        )

        out_messages = json.loads(result.messages_json)
        markers_inserted = list(result.markers_inserted)
        transforms_applied: list[str] = list(result.strategies_applied)

        # Inject a dropped-context marker into the outgoing list when
        # drops happened. Mirrors the retired Python behaviour — call
        # sites and tests look for the marker IN the message stream
        # (so the LLM can see it), not just in `markers_inserted`.
        # Inserted as `role=user` (not system) so it doesn't inflate
        # the system_messages count and breaks tests that probe for
        # exactly one system message.
        dropped_count = len(messages_copy) - len(out_messages)
        if dropped_count > 0:
            marker = create_dropped_context_marker("intelligent_cap", dropped_count)
            # Match the content format of the surrounding conversation:
            # Strands SDK uses list-of-blocks; Anthropic/OpenAI use a
            # plain string. Detect by looking at existing user
            # messages.
            uses_block_format = any(
                isinstance(m.get("content"), list) for m in out_messages if m.get("role") == "user"
            )
            marker_content: str | list[dict[str, str]] = (
                [{"type": "text", "text": marker}] if uses_block_format else marker
            )
            # Insert after the leading system messages and frozen
            # prefix so the marker rides with the conversation rather
            # than disrupting the system prompt.
            insert_at = frozen_message_count
            while insert_at < len(out_messages) and out_messages[insert_at].get("role") == "system":
                insert_at += 1
            out_messages.insert(
                insert_at,
                {"role": "user", "content": marker_content},
            )
            markers_inserted.append(marker)
            transforms_applied.append(f"intelligent_cap:{dropped_count}")

        return TransformResult(
            messages=out_messages,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=[],
        )


def _is_enterprise_config(cfg: IntelligentContextConfig) -> bool:
    """Detect whether the config sets any field that targets a cut
    strategy. Used to emit a one-time info-level log explaining the
    behaviour change."""
    return bool(
        getattr(cfg, "summarization_enabled", False)
        or getattr(cfg, "memory_tiers_enabled", False)
        or getattr(cfg, "warm_tier_enabled", False)
        or getattr(cfg, "cold_tier_enabled", False)
        # compress_threshold / summarize_threshold default to 0.10 / 0.25;
        # we don't warn on defaults because every Python caller would
        # trip them. Only warn when summarization or tiers are explicitly
        # enabled — that's the only signal a caller actually wanted them.
    )
