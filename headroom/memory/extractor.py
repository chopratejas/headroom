"""Memory extraction using LLMs.

Supports multiple providers by reusing the wrapped client with a cheap model.
Auto-detects provider from client class and selects appropriate cheap model.
Uses structured JSON output where available for reliable parsing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from headroom.memory.store import Memory

logger = logging.getLogger(__name__)


# Provider → Cheap Model mapping (verified January 2026)
# These are the most cost-effective models for simple extraction tasks
CHEAP_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",  # $0.15/1M input, $0.60/1M output
    "anthropic": "claude-3-5-haiku-latest",  # $0.80/1M input, $4/1M output
    "mistralai": "mistral-small-latest",  # $0.10/1M input, $0.30/1M output
    "groq": "llama-3.3-70b-versatile",  # Free tier available
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # $0.88/1M
    "fireworks": "accounts/fireworks/models/llama-v3p1-8b-instruct",  # $0.20/1M
    "google": "gemini-2.0-flash-lite",  # $0.075/1M input, $0.30/1M output
    "cohere": "command-r7b-12-2024",  # $0.0375/1M input, $0.15/1M output
}

# Providers that support structured JSON output via response_format
SUPPORTS_JSON_MODE: set[str] = {"openai", "mistralai", "groq", "together", "fireworks"}


# Entity-agnostic prompt - works for users, agents, or any conversational entity
EXTRACTION_PROMPT = """Analyze this conversation and extract any facts worth remembering.

Focus on:
- Preferences (language, tools, frameworks, style, configuration)
- Facts (identity, role, capabilities, constraints, environment)
- Context (goals, ongoing tasks, relationships, history)

Conversation:
Speaker A: {query}
Speaker B: {response}

Return a JSON object with this structure:
{{
  "memories": [
    {{"content": "Prefers Python for backend development", "category": "preference", "importance": 0.8}},
    {{"content": "Works on distributed systems", "category": "fact", "importance": 0.7}}
  ],
  "should_remember": true
}}

Categories: "preference", "fact", "context"
Importance: 0.0-1.0 (higher = more important to remember long-term)

If there's nothing worth remembering (greetings, generic questions, transient info), return:
{{"memories": [], "should_remember": false}}

Return ONLY valid JSON."""


class ChatClient(Protocol):
    """Protocol for chat clients (OpenAI, Anthropic, etc.)."""

    class Chat:
        class Completions:
            def create(self, **kwargs: Any) -> Any: ...

        completions: Completions

    chat: Chat


def detect_provider(client: Any) -> str | None:
    """Detect the provider from client class path.

    Args:
        client: The LLM client instance

    Returns:
        Provider name or None if unknown
    """
    module = type(client).__module__.lower()

    # Check for known providers
    providers = [
        "openai",
        "anthropic",
        "mistralai",
        "groq",
        "together",
        "fireworks",
        "google",
        "cohere",
    ]

    for provider in providers:
        if provider in module:
            return provider

    return None


def get_cheap_model(provider: str) -> str | None:
    """Get the cheap model for a provider.

    Args:
        provider: Provider name

    Returns:
        Cheap model ID or None if unknown
    """
    return CHEAP_MODELS.get(provider)


class MemoryExtractor:
    """Extracts memories from conversations using LLMs.

    Supports multiple providers by reusing the wrapped client.
    Auto-detects provider and selects appropriate cheap model.

    Usage:
        extractor = MemoryExtractor(openai_client)
        memories = extractor.extract("I prefer Python", "Great choice!")
    """

    def __init__(
        self,
        client: Any,
        model: str | None = None,
    ):
        """Initialize the extractor.

        Args:
            client: LLM client (OpenAI, Anthropic, etc.)
            model: Override the extraction model (auto-detects if None)
        """
        self.client = client
        self._provider = detect_provider(client)
        self._model: str | None = None

        if model:
            self._model = model
        elif self._provider:
            self._model = get_cheap_model(self._provider)

        if not self._model:
            logger.warning(
                f"Could not detect cheap model for provider. "
                f"Client type: {type(client).__module__}.{type(client).__name__}. "
                f"Memory extraction may fail."
            )

    @property
    def provider(self) -> str | None:
        """Get the detected provider."""
        return self._provider

    @property
    def model(self) -> str | None:
        """Get the extraction model."""
        return self._model

    def extract(self, query: str, response: str) -> list[Memory]:
        """Extract memories from a conversation turn.

        Args:
            query: User's message
            response: Assistant's response

        Returns:
            List of extracted memories (may be empty)
        """
        if not self._model:
            logger.warning("No extraction model configured, skipping extraction")
            return []

        prompt = EXTRACTION_PROMPT.format(query=query, response=response)

        try:
            result = self._call_llm(prompt)
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return []

    def extract_batch(self, conversations: list[tuple[str, str, str]]) -> dict[str, list[Memory]]:
        """Extract memories from multiple conversations.

        Args:
            conversations: List of (user_id, query, response) tuples

        Returns:
            Dict mapping user_id to list of memories
        """
        if not conversations:
            return {}

        # Build batch prompt
        batch_prompt = self._build_batch_prompt(conversations)

        try:
            result = self._call_llm(batch_prompt)
            return self._parse_batch_response(result, conversations)
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return {}

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt.

        Uses structured JSON output (response_format) where available
        to ensure reliable JSON parsing.

        Args:
            prompt: The prompt to send

        Returns:
            The LLM's response text
        """
        if self._provider == "anthropic":
            # Anthropic uses different API - no native JSON mode yet
            response = self.client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(response.content[0].text)
        elif self._provider == "cohere":
            # Cohere uses different API
            response = self.client.chat(
                model=self._model,
                message=prompt,
            )
            return str(response.text)
        elif self._provider == "google":
            # Google Gemini - use JSON response mime type
            model = self.client.GenerativeModel(
                self._model,
                generation_config={"response_mime_type": "application/json"},
            )
            response = model.generate_content(prompt)
            return str(response.text)
        else:
            # OpenAI-compatible API (OpenAI, Groq, Together, Fireworks, Mistral)
            # Use JSON mode for structured output
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic for extraction
            }

            # Add response_format for providers that support it
            if self._provider in SUPPORTS_JSON_MODE:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return str(response.choices[0].message.content)

    def _parse_response(self, text: str) -> list[Memory]:
        """Parse LLM response into memories.

        Args:
            text: Raw LLM response

        Returns:
            List of Memory objects
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

            data = json.loads(text.strip())

            if not data.get("should_remember", False):
                return []

            memories = []
            for item in data.get("memories", []):
                memories.append(
                    Memory(
                        content=item["content"],
                        category=item.get("category", "fact"),
                        importance=item.get("importance", 0.5),
                    )
                )

            return memories

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return []

    def _build_batch_prompt(self, conversations: list[tuple[str, str, str]]) -> str:
        """Build a batch extraction prompt.

        Args:
            conversations: List of (entity_id, query, response) tuples

        Returns:
            Batch prompt string
        """
        lines = [
            "Analyze these conversations and extract facts worth remembering about each entity.",
            "",
            "Focus on: preferences, facts, context that helps future interactions.",
            "",
        ]

        for i, (entity_id, query, response) in enumerate(conversations):
            lines.extend(
                [
                    f"--- Conversation {i + 1} (Entity: {entity_id}) ---",
                    f"Speaker A: {query}",
                    f"Speaker B: {response}",
                    "",
                ]
            )

        lines.extend(
            [
                "Return a JSON object mapping entity_id to their memories:",
                "{",
                '  "entity_123": {',
                '    "memories": [{"content": "...", "category": "preference", "importance": 0.8}],',
                '    "should_remember": true',
                "  }",
                "}",
                "",
                "Categories: preference, fact, context",
                "Importance: 0.0-1.0",
                "",
                "Return ONLY valid JSON.",
            ]
        )

        return "\n".join(lines)

    def _parse_batch_response(
        self,
        text: str,
        conversations: list[tuple[str, str, str]],
    ) -> dict[str, list[Memory]]:
        """Parse batch extraction response.

        Args:
            text: Raw LLM response
            conversations: Original conversations for fallback

        Returns:
            Dict mapping user_id to list of memories
        """
        try:
            # Extract JSON from response
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

            data = json.loads(text.strip())
            result: dict[str, list[Memory]] = {}

            for user_id, user_data in data.items():
                if not user_data.get("should_remember", False):
                    continue

                memories = []
                for item in user_data.get("memories", []):
                    memories.append(
                        Memory(
                            content=item["content"],
                            category=item.get("category", "fact"),
                            importance=item.get("importance", 0.5),
                        )
                    )

                if memories:
                    result[user_id] = memories

            return result

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to parse batch response: {e}")
            return {}
