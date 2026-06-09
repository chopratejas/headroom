"""Injectable seam Protocol types for HeadroomEngine.

These Protocols define the boundary between the engine facade and the
concrete subsystems (pipeline, CCR, TOIN).  Tests inject fakes; the
real server wires the concrete implementations in Chunk 4.

Only the minimal surface the facade actually calls is declared here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CompressionPipeline(Protocol):
    """Minimal protocol for a compression pipeline.

    Wraps ``TransformPipeline.apply`` — the facade calls *only* this method.
    The return value must be TransformResult-shaped: has ``.messages``,
    ``.tokens_before``, ``.tokens_after``, ``.transforms_applied``.

    Using a Protocol (not ABC) keeps the real ``TransformPipeline`` a
    zero-change dependency — it satisfies this interface structurally.
    """

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Apply compression transforms.

        Parameters
        ----------
        messages:
            List of conversation messages to transform.
        model:
            Model identifier (e.g. ``"claude-3-5-sonnet-20241022"``).
        **kwargs:
            Extra keyword arguments forwarded to the implementation
            (e.g. ``model_limit``, ``compression_policy``).

        Returns
        -------
        Any
            A TransformResult-shaped object with at minimum a ``.messages``
            attribute containing the (possibly mutated) message list.
        """
        ...
