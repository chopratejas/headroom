"""LangChain integration for Headroom SDK.

This module provides seamless integration with LangChain, enabling automatic
context optimization for any LangChain chat model.

Key insight: LangChain callbacks CANNOT modify messages (by design - see
https://github.com/langchain-ai/langchain/issues/8725). Therefore, we wrap
the chat model itself to intercept and transform messages.

Components:
1. HeadroomChatModel - Wraps any BaseChatModel to apply Headroom transforms
2. HeadroomCallbackHandler - Tracks metrics and token usage (observability only)
3. HeadroomRunnable - LCEL-compatible Runnable for chain composition
4. optimize_messages() - Standalone function for manual optimization

Example:
    from langchain_openai import ChatOpenAI
    from headroom.integrations import HeadroomChatModel

    # Wrap any LangChain chat model
    llm = ChatOpenAI(model="gpt-4o")
    optimized_llm = HeadroomChatModel(llm)

    # Use normally - Headroom automatically optimizes context
    response = optimized_llm.invoke("What is 2+2?")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

# LangChain imports - these are optional dependencies
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import RunnableLambda
    from pydantic import Field, PrivateAttr

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = object
    BaseCallbackHandler = object
    Field = lambda **kwargs: None  # noqa: E731
    PrivateAttr = lambda **kwargs: None  # noqa: E731

from headroom import HeadroomConfig, HeadroomMode
from headroom.providers import OpenAIProvider
from headroom.transforms import TransformPipeline

logger = logging.getLogger(__name__)


def _check_langchain_available():
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install headroom[langchain] "
            "or: pip install langchain-core langchain-openai"
        )


def langchain_available() -> bool:
    """Check if LangChain is installed."""
    return LANGCHAIN_AVAILABLE


@dataclass
class OptimizationMetrics:
    """Metrics from a single optimization pass."""

    request_id: str
    timestamp: datetime
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    transforms_applied: list[str]
    model: str


class HeadroomChatModel(BaseChatModel):
    """LangChain chat model wrapper that applies Headroom optimizations.

    Wraps any LangChain BaseChatModel and automatically optimizes the context
    before each API call. This is the recommended way to use Headroom with
    LangChain because:

    1. Callbacks cannot modify messages (LangChain design limitation)
    2. Wrapping ensures ALL calls go through optimization
    3. Works with streaming, tools, and all LangChain features

    Example:
        from langchain_openai import ChatOpenAI
        from headroom.integrations import HeadroomChatModel

        # Basic usage
        llm = ChatOpenAI(model="gpt-4o")
        optimized = HeadroomChatModel(llm)
        response = optimized.invoke([HumanMessage("Hello!")])

        # With custom config
        from headroom import HeadroomConfig, HeadroomMode
        config = HeadroomConfig(default_mode=HeadroomMode.OPTIMIZE)
        optimized = HeadroomChatModel(llm, config=config)

        # Access metrics
        print(f"Saved {optimized.total_tokens_saved} tokens")

    Attributes:
        wrapped_model: The underlying LangChain chat model
        headroom_client: HeadroomClient instance for optimization
        metrics_history: List of OptimizationMetrics from recent calls
        total_tokens_saved: Running total of tokens saved
    """

    # Pydantic model fields
    wrapped_model: Any = Field(description="The wrapped LangChain chat model")
    headroom_config: Any = Field(default=None, description="Headroom configuration")
    mode: HeadroomMode = Field(default=HeadroomMode.OPTIMIZE, description="Headroom mode")

    # Private attributes (not serialized)
    _metrics_history: list = PrivateAttr(default_factory=list)
    _total_tokens_saved: int = PrivateAttr(default=0)
    _pipeline: Any = PrivateAttr(default=None)
    _provider: Any = PrivateAttr(default=None)

    class Config:
        """Pydantic config for LangChain compatibility."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        wrapped_model: BaseChatModel,
        config: HeadroomConfig | None = None,
        mode: HeadroomMode = HeadroomMode.OPTIMIZE,
        **kwargs,
    ):
        """Initialize HeadroomChatModel.

        Args:
            wrapped_model: Any LangChain BaseChatModel to wrap
            config: HeadroomConfig for optimization settings
            mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
            **kwargs: Additional arguments passed to BaseChatModel
        """
        _check_langchain_available()

        super().__init__(
            wrapped_model=wrapped_model,
            headroom_config=config or HeadroomConfig(),
            mode=mode,
            **kwargs,
        )
        self._metrics_history = []
        self._total_tokens_saved = 0
        self._pipeline = None
        self._provider = None

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return f"headroom-{self.wrapped_model._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters."""
        return {
            "wrapped_model": self.wrapped_model._identifying_params,
            "headroom_mode": self.mode.value,
        }

    @property
    def pipeline(self) -> TransformPipeline:
        """Lazily initialize TransformPipeline."""
        if self._pipeline is None:
            self._provider = OpenAIProvider()
            self._pipeline = TransformPipeline(
                config=self.headroom_config,
                provider=self._provider,
            )
        return self._pipeline

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls."""
        return self._total_tokens_saved

    @property
    def metrics_history(self) -> list[OptimizationMetrics]:
        """History of optimization metrics."""
        return self._metrics_history.copy()

    def _convert_messages_to_openai(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to OpenAI format for Headroom."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                # Generic fallback
                result.append(
                    {
                        "role": getattr(msg, "type", "user"),
                        "content": msg.content,
                    }
                )
        return result

    def _convert_messages_from_openai(self, messages: list[dict[str, Any]]) -> list[BaseMessage]:
        """Convert OpenAI format messages back to LangChain format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                tool_calls = []
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_calls.append(
                            {
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "args": json.loads(tc["function"]["arguments"]),
                            }
                        )
                result.append(AIMessage(content=content, tool_calls=tool_calls))
            elif role == "tool":
                result.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
        return result

    def _optimize_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], OptimizationMetrics]:
        """Apply Headroom optimization to messages."""
        request_id = str(uuid4())

        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai(messages)

        # Get model name and context limit
        model = getattr(self.wrapped_model, "model_name", None)
        if model is None:
            model = getattr(self.wrapped_model, "model", "gpt-4o")

        # Get model context limit from provider
        model_limit = self._provider.get_context_limit(model) if self._provider else 128000

        # Apply Headroom transforms via pipeline
        result = self.pipeline.apply(
            messages=openai_messages,
            model=model,
            model_limit=model_limit,
        )

        # Create metrics
        metrics = OptimizationMetrics(
            request_id=request_id,
            timestamp=datetime.now(),
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=result.tokens_before - result.tokens_after,
            savings_percent=(
                (result.tokens_before - result.tokens_after) / result.tokens_before * 100
                if result.tokens_before > 0
                else 0
            ),
            transforms_applied=result.transforms_applied,
            model=model,
        )

        # Track metrics
        self._metrics_history.append(metrics)
        self._total_tokens_saved += metrics.tokens_saved

        # Keep only last 100 metrics
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]

        # Convert back to LangChain format
        optimized_messages = self._convert_messages_from_openai(result.messages)

        return optimized_messages, metrics

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs,
    ) -> ChatResult:
        """Generate response with Headroom optimization.

        This is the core method called by invoke(), batch(), etc.
        """
        # Optimize messages
        optimized_messages, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized: {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.savings_percent:.1f}% saved)"
        )

        # Call wrapped model with optimized messages
        result = self.wrapped_model._generate(
            optimized_messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs,
    ) -> Iterator[ChatGeneration]:
        """Stream response with Headroom optimization."""
        # Optimize messages
        optimized_messages, metrics = self._optimize_messages(messages)

        logger.info(
            f"Headroom optimized (streaming): {metrics.tokens_before} -> {metrics.tokens_after} tokens"
        )

        # Stream from wrapped model
        yield from self.wrapped_model._stream(
            optimized_messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs) -> HeadroomChatModel:
        """Bind tools to the wrapped model."""
        new_wrapped = self.wrapped_model.bind_tools(tools, **kwargs)
        return HeadroomChatModel(
            wrapped_model=new_wrapped,
            config=self.headroom_config,
            mode=self.mode,
        )

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings."""
        if not self._metrics_history:
            return {
                "total_requests": 0,
                "total_tokens_saved": 0,
                "average_savings_percent": 0,
            }

        return {
            "total_requests": len(self._metrics_history),
            "total_tokens_saved": self._total_tokens_saved,
            "average_savings_percent": sum(m.savings_percent for m in self._metrics_history)
            / len(self._metrics_history),
            "total_tokens_before": sum(m.tokens_before for m in self._metrics_history),
            "total_tokens_after": sum(m.tokens_after for m in self._metrics_history),
        }


class HeadroomCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Headroom metrics and observability.

    NOTE: Callbacks CANNOT modify messages in LangChain (by design).
    Use HeadroomChatModel for actual optimization. This handler is for:

    1. Tracking token usage across chains
    2. Logging optimization metrics
    3. Alerting on high token usage
    4. Integration with observability platforms

    Example:
        from langchain_openai import ChatOpenAI
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler(
            log_level="INFO",
            token_alert_threshold=10000,
        )

        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        response = llm.invoke("Hello!")

        # Check metrics
        print(f"Total tokens: {handler.total_tokens}")
        print(f"Alerts: {handler.alerts}")
    """

    def __init__(
        self,
        log_level: str = "INFO",
        token_alert_threshold: int | None = None,
        cost_alert_threshold: float | None = None,
    ):
        """Initialize callback handler.

        Args:
            log_level: Logging level for metrics ("DEBUG", "INFO", "WARNING")
            token_alert_threshold: Alert if request exceeds this many tokens
            cost_alert_threshold: Alert if estimated cost exceeds this amount
        """
        _check_langchain_available()

        self.log_level = log_level
        self.token_alert_threshold = token_alert_threshold
        self.cost_alert_threshold = cost_alert_threshold

        # Metrics tracking
        self._requests: list[dict[str, Any]] = []
        self._total_tokens = 0
        self._alerts: list[str] = []
        self._current_request: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all requests."""
        return self._total_tokens

    @property
    def total_requests(self) -> int:
        """Total number of requests tracked."""
        return len(self._requests)

    @property
    def alerts(self) -> list[str]:
        """List of alerts triggered."""
        return self._alerts.copy()

    @property
    def requests(self) -> list[dict[str, Any]]:
        """List of request metrics."""
        return self._requests.copy()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs,
    ) -> None:
        """Called when LLM starts processing."""
        self._current_request = {
            "start_time": datetime.now(),
            "model": serialized.get("name", "unknown"),
            "prompt_count": len(prompts),
            "estimated_input_tokens": sum(len(p) // 4 for p in prompts),  # Rough estimate
        }

        if self.log_level == "DEBUG":
            logger.debug(f"LLM request started: {self._current_request}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs,
    ) -> None:
        """Called when chat model starts processing."""
        # Estimate tokens from messages
        total_content = ""
        for msg_list in messages:
            for msg in msg_list:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                total_content += content

        estimated_tokens = len(total_content) // 4  # Rough estimate

        self._current_request = {
            "start_time": datetime.now(),
            "model": serialized.get("name", serialized.get("id", ["unknown"])[-1]),
            "message_count": sum(len(ml) for ml in messages),
            "estimated_input_tokens": estimated_tokens,
        }

        # Check token alert
        if self.token_alert_threshold and estimated_tokens > self.token_alert_threshold:
            alert = f"Token alert: {estimated_tokens} tokens exceeds threshold {self.token_alert_threshold}"
            self._alerts.append(alert)
            logger.warning(alert)

        if self.log_level in ("DEBUG", "INFO"):
            logger.log(
                logging.DEBUG if self.log_level == "DEBUG" else logging.INFO,
                f"Chat model request: ~{estimated_tokens} input tokens",
            )

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM finishes processing."""
        if self._current_request is None:
            return

        # Extract token usage from response if available
        token_usage = {}
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        self._current_request["end_time"] = datetime.now()
        self._current_request["duration_ms"] = (
            self._current_request["end_time"] - self._current_request["start_time"]
        ).total_seconds() * 1000

        if token_usage:
            self._current_request["input_tokens"] = token_usage.get("prompt_tokens", 0)
            self._current_request["output_tokens"] = token_usage.get("completion_tokens", 0)
            self._current_request["total_tokens"] = token_usage.get("total_tokens", 0)
            self._total_tokens += self._current_request["total_tokens"]

        self._requests.append(self._current_request)

        # Keep only last 1000 requests
        if len(self._requests) > 1000:
            self._requests = self._requests[-1000:]

        if self.log_level in ("DEBUG", "INFO"):
            tokens_info = f"{self._current_request.get('total_tokens', 'unknown')} tokens"
            duration = f"{self._current_request['duration_ms']:.0f}ms"
            logger.log(
                logging.DEBUG if self.log_level == "DEBUG" else logging.INFO,
                f"LLM request completed: {tokens_info} in {duration}",
            )

        self._current_request = None

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        if self._current_request:
            self._current_request["error"] = str(error)
            self._current_request["end_time"] = datetime.now()
            self._requests.append(self._current_request)
            self._current_request = None

        logger.error(f"LLM error: {error}")

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all tracked requests."""
        if not self._requests:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "average_tokens": 0,
                "average_duration_ms": 0,
                "errors": 0,
                "alerts": len(self._alerts),
            }

        successful = [r for r in self._requests if "error" not in r]
        total_tokens = sum(r.get("total_tokens", 0) for r in successful)

        return {
            "total_requests": len(self._requests),
            "successful_requests": len(successful),
            "total_tokens": total_tokens,
            "average_tokens": total_tokens / len(successful) if successful else 0,
            "average_duration_ms": (
                sum(r.get("duration_ms", 0) for r in successful) / len(successful)
                if successful
                else 0
            ),
            "errors": len(self._requests) - len(successful),
            "alerts": len(self._alerts),
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._requests = []
        self._total_tokens = 0
        self._alerts = []
        self._current_request = None


class HeadroomRunnable:
    """LCEL-compatible Runnable for Headroom optimization.

    Use this to add Headroom optimization to any LangChain chain using LCEL.

    Example:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from headroom.integrations import HeadroomRunnable

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-4o")

        # Add Headroom optimization to chain
        chain = prompt | HeadroomRunnable() | llm
        response = chain.invoke({"input": "Hello!"})
    """

    def __init__(
        self,
        config: HeadroomConfig | None = None,
        mode: HeadroomMode = HeadroomMode.OPTIMIZE,
    ):
        """Initialize HeadroomRunnable.

        Args:
            config: HeadroomConfig for optimization settings
            mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
        """
        _check_langchain_available()

        self.config = config or HeadroomConfig()
        self.mode = mode
        self._pipeline: TransformPipeline | None = None
        self._provider: OpenAIProvider | None = None
        self._metrics_history: list[OptimizationMetrics] = []

    @property
    def pipeline(self) -> TransformPipeline:
        """Lazily initialize TransformPipeline."""
        if self._pipeline is None:
            self._provider = OpenAIProvider()
            self._pipeline = TransformPipeline(
                config=self.config,
                provider=self._provider,
            )
        return self._pipeline

    def __or__(self, other):
        """Support pipe operator for LCEL composition."""
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence(first=self.as_runnable(), last=other)

    def __ror__(self, other):
        """Support reverse pipe operator."""
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence(first=other, last=self.as_runnable())

    def as_runnable(self):
        """Convert to LangChain Runnable."""
        return RunnableLambda(self._optimize)

    def _optimize(self, input_data: Any) -> Any:
        """Optimize input messages."""
        # Handle different input types
        if isinstance(input_data, list):
            messages = input_data
        elif hasattr(input_data, "messages"):
            messages = input_data.messages
        elif hasattr(input_data, "to_messages"):
            messages = input_data.to_messages()
        else:
            # Can't optimize, pass through
            return input_data

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                openai_messages.append(
                    {
                        "role": msg.type,
                        "content": msg.content,
                    }
                )

        # Get model context limit
        model = "gpt-4o"  # Default model for estimation
        model_limit = self._provider.get_context_limit(model) if self._provider else 128000

        # Apply Headroom transforms via pipeline
        result = self.pipeline.apply(
            messages=openai_messages,
            model=model,
            model_limit=model_limit,
        )

        # Track metrics
        metrics = OptimizationMetrics(
            request_id=str(uuid4()),
            timestamp=datetime.now(),
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=result.tokens_before - result.tokens_after,
            savings_percent=(
                (result.tokens_before - result.tokens_after) / result.tokens_before * 100
                if result.tokens_before > 0
                else 0
            ),
            transforms_applied=result.transforms_applied,
            model="gpt-4o",
        )
        self._metrics_history.append(metrics)

        # Convert back to LangChain messages
        output_messages = []
        for msg in result.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                output_messages.append(SystemMessage(content=content))
            elif role == "user":
                output_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                output_messages.append(AIMessage(content=content))
            elif role == "tool":
                output_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )

        return output_messages


def optimize_messages(
    messages: list[BaseMessage],
    config: HeadroomConfig | None = None,
    mode: HeadroomMode = HeadroomMode.OPTIMIZE,
    model: str = "gpt-4o",
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Standalone function to optimize LangChain messages.

    Use this for manual optimization when you need fine-grained control.

    Args:
        messages: List of LangChain BaseMessage objects
        config: HeadroomConfig for optimization settings
        mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
        model: Model name for token estimation

    Returns:
        Tuple of (optimized_messages, metrics_dict)

    Example:
        from langchain_core.messages import HumanMessage, SystemMessage
        from headroom.integrations import optimize_messages

        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="What is 2+2?"),
        ]

        optimized, metrics = optimize_messages(messages)
        print(f"Saved {metrics['tokens_saved']} tokens")
    """
    _check_langchain_available()

    config = config or HeadroomConfig()
    provider = OpenAIProvider()
    pipeline = TransformPipeline(config=config, provider=provider)

    # Convert to OpenAI format
    openai_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry = {"role": "assistant", "content": msg.content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            openai_messages.append(entry)
        elif isinstance(msg, ToolMessage):
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )

    # Get model context limit
    model_limit = provider.get_context_limit(model)

    # Apply transforms via pipeline
    result = pipeline.apply(
        messages=openai_messages,
        model=model,
        model_limit=model_limit,
    )

    # Convert back
    output_messages = []
    for msg in result.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            output_messages.append(SystemMessage(content=content))
        elif role == "user":
            output_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = []
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_calls.append(
                        {
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]),
                        }
                    )
            output_messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif role == "tool":
            output_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                )
            )

    metrics = {
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_before - result.tokens_after,
        "savings_percent": (
            (result.tokens_before - result.tokens_after) / result.tokens_before * 100
            if result.tokens_before > 0
            else 0
        ),
        "transforms_applied": result.transforms_applied,
    }

    return output_messages, metrics
