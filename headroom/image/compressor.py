"""Image Compressor - Seamless image token optimization.

This is the main entry point for image compression in Headroom.
It automatically:
1. Detects images in messages
2. Extracts the user's query
3. Routes to optimal compression technique (via trained model)
4. Applies provider-specific compression

Usage:
    from headroom.image import ImageCompressor

    compressor = ImageCompressor()

    # Compress images in a request
    compressed = compressor.compress(messages, provider="openai")

    # Check savings
    print(f"Saved {compressor.last_savings}% tokens")
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trained_router import TrainedRouter

from .trained_router import Technique

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of image compression."""

    technique: Technique
    original_tokens: int
    compressed_tokens: int
    confidence: float

    @property
    def savings_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compressed_tokens / self.original_tokens) * 100


class ImageCompressor:
    """Seamless image compression for LLM requests.

    Automatically detects images, analyzes queries, and applies
    optimal compression based on a trained ML model.

    The model is downloaded from HuggingFace on first use:
    https://huggingface.co/chopratejas/technique-router

    Args:
        model_id: HuggingFace model ID (default: chopratejas/technique-router)
        use_siglip: Whether to use SigLIP for image analysis (default: True)
        device: Device for inference ('cuda', 'cpu', or None for auto)
    """

    def __init__(
        self,
        model_id: str | None = None,
        use_siglip: bool = True,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.use_siglip = use_siglip
        self.device = device

        # Lazy-loaded router
        self._router: TrainedRouter | None = None

        # Last compression result (for metrics)
        self.last_result: CompressionResult | None = None

    @property
    def last_savings(self) -> float:
        """Savings from last compression (percentage)."""
        if self.last_result:
            return self.last_result.savings_percent
        return 0.0

    def _get_router(self) -> TrainedRouter:
        """Lazy load the trained router."""
        if self._router is None:
            from .trained_router import TrainedRouter

            self._router = TrainedRouter(
                model_path=self.model_id,
                use_siglip=self.use_siglip,
                device=self.device,
            )
        return self._router

    def has_images(self, messages: list[dict[str, Any]]) -> bool:
        """Check if messages contain images."""
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # OpenAI format
                        if item.get("type") == "image_url":
                            return True
                        # Anthropic format
                        if item.get("type") == "image":
                            return True
                        # Google format
                        if "inlineData" in item:
                            return True
        return False

    def _extract_query(self, messages: list[dict[str, Any]]) -> str:
        """Extract the text query from messages."""
        # Look for user message with text
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content")

            # Simple string content
            if isinstance(content, str):
                return content

            # Multi-part content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
                if texts:
                    return " ".join(texts)

        return ""

    def _extract_image_data(self, messages: list[dict[str, Any]]) -> bytes | None:
        """Extract first image data from messages."""
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                # OpenAI format: {"type": "image_url", "image_url": {"url": "data:..."}}
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        # Extract base64 data
                        match = re.match(r"data:image/[^;]+;base64,(.+)", url)
                        if match:
                            return base64.b64decode(match.group(1))

                # Anthropic format: {"type": "image", "source": {"data": "..."}}
                if item.get("type") == "image":
                    source = item.get("source", {})
                    if source.get("type") == "base64":
                        return base64.b64decode(source.get("data", ""))

                # Google format: {"inlineData": {"data": "..."}}
                if "inlineData" in item:
                    return base64.b64decode(item["inlineData"].get("data", ""))

        return None

    def _resize_image(
        self, image_data: bytes, max_dimension: int = 512, quality: int = 85
    ) -> tuple[bytes, str]:
        """Resize image to reduce tokens.

        Args:
            image_data: Original image bytes
            max_dimension: Maximum width or height
            quality: JPEG quality (1-100)

        Returns:
            Tuple of (resized_bytes, media_type)
        """
        from PIL import Image

        img = Image.open(io.BytesIO(image_data))
        original_format = img.format or "PNG"

        # Calculate new dimensions preserving aspect ratio
        width, height = img.size
        if width <= max_dimension and height <= max_dimension:
            # Already small enough
            return image_data, f"image/{original_format.lower()}"

        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        # Resize
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to RGB if needed (for JPEG)
        if resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        # Save as JPEG for best compression
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg"

    def _estimate_tokens(self, image_data: bytes, detail: str = "high") -> int:
        """Estimate token count for image (OpenAI formula)."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
        except Exception:
            # Default estimate
            return 765

        if detail == "low":
            return 85

        # High detail: 85 tokens per 512x512 tile + 170 base
        tiles_x = (width + 511) // 512
        tiles_y = (height + 511) // 512
        return int(85 * tiles_x * tiles_y + 170)

    def _apply_compression(
        self,
        messages: list[dict[str, Any]],
        technique: Technique,
        provider: str,
    ) -> list[dict[str, Any]]:
        """Apply compression technique to messages."""
        if technique.value == "preserve":
            return messages

        compressed = []
        for message in messages:
            content = message.get("content")

            if not isinstance(content, list):
                compressed.append(message)
                continue

            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue

                # OpenAI format - compare by value since technique may be from trained_router
                if item.get("type") == "image_url" and provider == "openai":
                    if technique.value == "full_low":
                        # Apply detail="low"
                        new_item = {
                            "type": "image_url",
                            "image_url": {
                                **item.get("image_url", {}),
                                "detail": "low",
                            },
                        }
                        new_content.append(new_item)
                    elif technique.value == "crop":
                        # For now, use low detail (TODO: implement actual cropping)
                        new_item = {
                            "type": "image_url",
                            "image_url": {
                                **item.get("image_url", {}),
                                "detail": "low",
                            },
                        }
                        new_content.append(new_item)
                    elif technique.value == "transcode":
                        # TODO: Convert to text description
                        # For now, keep original
                        new_content.append(item)
                    else:
                        new_content.append(item)

                # Anthropic format - resize image for compression
                elif item.get("type") == "image" and provider == "anthropic":
                    if technique.value in ("full_low", "crop"):
                        # Resize image to reduce tokens
                        try:
                            source = item.get("source", {})
                            if source.get("type") == "base64":
                                original_data = base64.b64decode(source.get("data", ""))
                                resized_data, media_type = self._resize_image(
                                    original_data, max_dimension=512
                                )
                                new_item = {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64.b64encode(resized_data).decode(),
                                    },
                                }
                                new_content.append(new_item)
                            else:
                                new_content.append(item)
                        except Exception as e:
                            logger.warning(f"Failed to resize Anthropic image: {e}")
                            new_content.append(item)
                    else:
                        new_content.append(item)

                # Google format - resize image for compression
                elif "inlineData" in item and provider == "google":
                    if technique.value in ("full_low", "crop"):
                        try:
                            inline = item.get("inlineData", {})
                            original_data = base64.b64decode(inline.get("data", ""))
                            resized_data, media_type = self._resize_image(
                                original_data,
                                max_dimension=768,  # Google uses 768x768 tiles
                            )
                            new_item = {
                                "inlineData": {
                                    "mimeType": media_type,
                                    "data": base64.b64encode(resized_data).decode(),
                                }
                            }
                            new_content.append(new_item)
                        except Exception as e:
                            logger.warning(f"Failed to resize Google image: {e}")
                            new_content.append(item)
                    else:
                        new_content.append(item)

                else:
                    new_content.append(item)

            compressed.append({**message, "content": new_content})

        return compressed

    def compress(
        self,
        messages: list[dict[str, Any]],
        provider: str = "openai",
    ) -> list[dict[str, Any]]:
        """Compress images in messages.

        Args:
            messages: LLM messages (OpenAI/Anthropic/Google format)
            provider: Target provider ('openai', 'anthropic', 'google')

        Returns:
            Messages with compressed images
        """
        if not self.has_images(messages):
            return messages

        # Extract query and image
        query = self._extract_query(messages)
        image_data = self._extract_image_data(messages)

        if not query or not image_data:
            logger.debug("Could not extract query or image, skipping compression")
            return messages

        # Route to technique
        try:
            router = self._get_router()
            decision = router.classify(image_data, query)
            technique = decision.technique
            confidence = decision.confidence
        except Exception as e:
            logger.warning(f"Router failed, preserving image: {e}")
            technique = Technique.PRESERVE
            confidence = 0.0

        # Calculate tokens - compare by value since technique is from trained_router
        original_tokens = self._estimate_tokens(image_data, "high")

        if technique.value == "full_low":
            compressed_tokens = 85  # OpenAI low detail
        elif technique.value == "preserve":
            compressed_tokens = original_tokens
        elif technique.value == "crop":
            compressed_tokens = 85  # Approximation
        elif technique.value == "transcode":
            compressed_tokens = 50  # Text description estimate
        else:
            compressed_tokens = original_tokens

        # Store result
        self.last_result = CompressionResult(
            technique=technique,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            confidence=confidence,
        )

        logger.info(
            f"Image compression: {technique.value} "
            f"({original_tokens} → {compressed_tokens} tokens, "
            f"{self.last_result.savings_percent:.0f}% saved)"
        )

        # Apply compression
        return self._apply_compression(messages, technique, provider)


# Singleton for convenience
_default_compressor: ImageCompressor | None = None


def get_compressor() -> ImageCompressor:
    """Get the default ImageCompressor instance."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = ImageCompressor()
    return _default_compressor


def compress_images(
    messages: list[dict[str, Any]],
    provider: str = "openai",
) -> list[dict[str, Any]]:
    """Convenience function to compress images in messages.

    Args:
        messages: LLM messages
        provider: Target provider

    Returns:
        Messages with compressed images
    """
    return get_compressor().compress(messages, provider)
