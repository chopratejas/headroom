"""Transform pipeline orchestration for Headroom SDK."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..config import (
    CacheAlignerConfig,
    DiffArtifact,
    HeadroomConfig,
    RollingWindowConfig,
    ToolCrusherConfig,
    TransformDiff,
    TransformResult,
)
from ..tokenizer import Tokenizer
from ..utils import deep_copy_messages
from .base import Transform
from .cache_aligner import CacheAligner
from .rolling_window import RollingWindow
from .smart_crusher import SmartCrusher
from .tool_crusher import ToolCrusher

if TYPE_CHECKING:
    from ..providers.base import Provider

logger = logging.getLogger(__name__)


class TransformPipeline:
    """
    Orchestrates multiple transforms in the correct order.

    Transform order:
    1. Cache Aligner - normalize prefix for cache hits
    2. Tool Crusher - compress tool outputs
    3. Rolling Window - enforce token limits
    """

    def __init__(
        self,
        config: HeadroomConfig | None = None,
        transforms: list[Transform] | None = None,
        provider: Provider | None = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Headroom configuration.
            transforms: Optional custom transform list (overrides config).
            provider: Provider for model-specific behavior.
        """
        self.config = config or HeadroomConfig()
        self._provider = provider

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._build_default_transforms()

    def _build_default_transforms(self) -> list[Transform]:
        """Build default transform pipeline from config."""
        transforms: list[Transform] = []

        # Order matters!

        # 1. Cache Aligner (prefix stabilization)
        if self.config.cache_aligner.enabled:
            transforms.append(CacheAligner(self.config.cache_aligner))

        # 2. Tool Output Compression
        # SmartCrusher (statistical) takes precedence over ToolCrusher (fixed rules)
        if self.config.smart_crusher.enabled:
            # Use smart statistical crushing
            from .smart_crusher import SmartCrusherConfig as SCConfig

            smart_config = SCConfig(
                enabled=True,
                min_items_to_analyze=self.config.smart_crusher.min_items_to_analyze,
                min_tokens_to_crush=self.config.smart_crusher.min_tokens_to_crush,
                variance_threshold=self.config.smart_crusher.variance_threshold,
                uniqueness_threshold=self.config.smart_crusher.uniqueness_threshold,
                similarity_threshold=self.config.smart_crusher.similarity_threshold,
                max_items_after_crush=self.config.smart_crusher.max_items_after_crush,
                preserve_change_points=self.config.smart_crusher.preserve_change_points,
                factor_out_constants=self.config.smart_crusher.factor_out_constants,
                include_summaries=self.config.smart_crusher.include_summaries,
            )
            transforms.append(SmartCrusher(smart_config))
        elif self.config.tool_crusher.enabled:
            # Fallback to fixed-rule crushing
            transforms.append(ToolCrusher(self.config.tool_crusher))

        # 3. Rolling Window (enforce limits last)
        if self.config.rolling_window.enabled:
            transforms.append(RollingWindow(self.config.rolling_window))

        return transforms

    def _get_tokenizer(self, model: str) -> Tokenizer:
        """Get tokenizer for model using provider."""
        if self._provider is None:
            raise ValueError(
                "Provider is required for token counting. "
                "Pass a provider to TransformPipeline or HeadroomClient."
            )
        token_counter = self._provider.get_token_counter(model)
        return Tokenizer(token_counter, model)

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Apply all transforms in sequence.

        Args:
            messages: List of messages to transform.
            model: Model name for token counting.
            **kwargs: Additional arguments passed to transforms.
                - model_limit: Context limit override.
                - output_buffer: Output buffer override.
                - tool_profiles: Per-tool compression profiles.
                - request_id: Optional request ID for diff artifact.

        Returns:
            Combined TransformResult.
        """
        tokenizer = self._get_tokenizer(model)

        # Get model limit from kwargs (should be set by client)
        model_limit = kwargs.get("model_limit")
        if model_limit is None:
            raise ValueError(
                "model_limit is required. Provide it via kwargs or "
                "configure model_context_limits in HeadroomClient."
            )

        # Start with original tokens
        tokens_before = tokenizer.count_messages(messages)

        logger.debug(
            "Pipeline starting: %d messages, %d tokens, model=%s",
            len(messages),
            tokens_before,
            model,
        )

        # Track all transforms applied
        all_transforms: list[str] = []
        all_markers: list[str] = []
        all_warnings: list[str] = []

        # Track transform diffs if enabled
        transform_diffs: list[TransformDiff] = []
        generate_diff = self.config.generate_diff_artifact

        current_messages = deep_copy_messages(messages)

        for transform in self.transforms:
            # Check if transform should run
            if not transform.should_apply(current_messages, tokenizer, **kwargs):
                continue

            # Track tokens before this transform (for diff)
            tokens_before_transform = tokenizer.count_messages(current_messages)

            # Apply transform
            result = transform.apply(current_messages, tokenizer, **kwargs)

            # Update messages for next transform
            current_messages = result.messages

            # Track tokens after this transform (for diff)
            tokens_after_transform = tokenizer.count_messages(current_messages)

            # Accumulate results
            all_transforms.extend(result.transforms_applied)
            all_markers.extend(result.markers_inserted)
            all_warnings.extend(result.warnings)

            # Log transform results
            if result.transforms_applied:
                logger.info(
                    "Transform %s: %d -> %d tokens (saved %d)",
                    transform.name,
                    tokens_before_transform,
                    tokens_after_transform,
                    tokens_before_transform - tokens_after_transform,
                )
            else:
                logger.debug("Transform %s: no changes", transform.name)

            # Record diff if enabled
            if generate_diff:
                transform_diffs.append(
                    TransformDiff(
                        transform_name=transform.name,
                        tokens_before=tokens_before_transform,
                        tokens_after=tokens_after_transform,
                        tokens_saved=tokens_before_transform - tokens_after_transform,
                        details=", ".join(result.transforms_applied)
                        if result.transforms_applied
                        else "",
                    )
                )

        # Final token count
        tokens_after = tokenizer.count_messages(current_messages)

        # Log pipeline summary
        total_saved = tokens_before - tokens_after
        if total_saved > 0:
            logger.info(
                "Pipeline complete: %d -> %d tokens (saved %d, %.1f%% reduction)",
                tokens_before,
                tokens_after,
                total_saved,
                (total_saved / tokens_before * 100) if tokens_before > 0 else 0,
            )
        else:
            logger.debug("Pipeline complete: no token savings")

        # Build diff artifact if enabled
        diff_artifact = None
        if generate_diff:
            diff_artifact = DiffArtifact(
                request_id=kwargs.get("request_id", ""),
                original_tokens=tokens_before,
                optimized_tokens=tokens_after,
                total_tokens_saved=tokens_before - tokens_after,
                transforms=transform_diffs,
            )

        return TransformResult(
            messages=current_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=all_transforms,
            markers_inserted=all_markers,
            warnings=all_warnings,
            diff_artifact=diff_artifact,
        )

    def simulate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Simulate transforms without modifying messages.

        Same as apply() but returns what WOULD happen.

        Args:
            messages: List of messages.
            model: Model name.
            **kwargs: Additional arguments.

        Returns:
            TransformResult with simulated changes.
        """
        # apply() already works on a copy, so this is safe
        return self.apply(messages, model, **kwargs)


def create_pipeline(
    tool_crusher_config: ToolCrusherConfig | None = None,
    cache_aligner_config: CacheAlignerConfig | None = None,
    rolling_window_config: RollingWindowConfig | None = None,
) -> TransformPipeline:
    """
    Create a pipeline with specific configurations.

    Args:
        tool_crusher_config: Tool crusher configuration.
        cache_aligner_config: Cache aligner configuration.
        rolling_window_config: Rolling window configuration.

    Returns:
        Configured TransformPipeline.
    """
    config = HeadroomConfig()

    if tool_crusher_config is not None:
        config.tool_crusher = tool_crusher_config
    if cache_aligner_config is not None:
        config.cache_aligner = cache_aligner_config
    if rolling_window_config is not None:
        config.rolling_window = rolling_window_config

    return TransformPipeline(config)
