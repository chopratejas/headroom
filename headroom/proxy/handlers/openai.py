"""OpenAI handler mixin for HeadroomProxy.

Contains all OpenAI Chat Completions, Responses API, and passthrough handlers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request, WebSocket
    from fastapi.responses import JSONResponse, Response, StreamingResponse


logger = logging.getLogger("headroom.proxy")


class OpenAIHandlerMixin:
    """Mixin providing OpenAI API handler methods for HeadroomProxy."""

    async def handle_openai_chat(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/chat/completions endpoint."""
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import CCRToolInjector
        from headroom.proxy.helpers import (
            COMPRESSION_TIMEOUT_SECONDS,
            MAX_MESSAGE_ARRAY_LENGTH,
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        # Validate message array size
        if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Message array too large ({len(messages)} messages). "
                        f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                    }
                },
            )

        stream = body.get("stream", False)

        # Bypass: skip ALL compression for explicit opt-out
        _bypass = (
            request.headers.get("x-headroom-bypass", "").lower() == "true"
            or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
        )
        if _bypass:
            logger.info(f"[{request_id}] Bypass: skipping compression (header)")

        # TODO: Re-enable image compression once token counting is accurate.
        # See anthropic.py handler for details on why this is disabled.
        #
        # if self.config.image_optimize and messages and not _bypass:
        #     compressor = _get_image_compressor()
        #     if compressor and compressor.has_images(messages):
        #         messages = compressor.compress(messages, provider="openai")
        #         if compressor.last_result:
        #             logger.info(
        #                 f"Image compression: {compressor.last_result.technique.value} "
        #                 f"({compressor.last_result.savings_percent:.0f}% saved, "
        #                 f"{compressor.last_result.original_tokens} -> "
        #                 f"{compressor.last_result.compressed_tokens} tokens)"
        #             )

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Check cache
        if self.cache and not stream:
            cached = await self.cache.get(messages, model)
            if cached:
                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=0,  # Savings already counted when response was cached
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )

                # Remove compression headers from cached response
                response_headers = dict(cached.response_headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(content=cached.response_body, headers=response_headers)

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Hook: pre_compress
        _hook_biases = None
        if self.config.hooks:
            from headroom.hooks import CompressContext

            _hook_ctx = CompressContext(model=model, provider="openai")
            try:
                messages = self.config.hooks.pre_compress(messages, _hook_ctx)
                _hook_biases = self.config.hooks.compute_biases(messages, _hook_ctx)
            except Exception as e:
                logger.debug(f"[{request_id}] Hook error: {e}")

        # Optimization
        transforms_applied = []
        pipeline_timing: dict[str, float] = {}
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        # Get prefix cache tracker for this session
        openai_session_id = self.session_tracker_store.compute_session_id(request, model, messages)
        openai_prefix_tracker = self.session_tracker_store.get_or_create(
            openai_session_id, "openai"
        )
        openai_frozen_count = openai_prefix_tracker.get_frozen_message_count()

        _compression_failed = False
        original_messages = messages  # Preserve for 400-retry fallback
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and not _bypass and _license_ok:
            try:
                context_limit = self.openai_provider.get_context_limit(model)

                if self.config.mode == "token_headroom":
                    comp_cache = self._get_compression_cache(openai_session_id)

                    # Zone 1: Swap cached compressed versions
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary
                    openai_frozen_count = comp_cache.compute_frozen_count(messages)

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.openai_pipeline.apply(
                                messages=working_messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(working_messages),
                                frozen_message_count=openai_frozen_count,
                                biases=_hook_biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)

                    # Always use pipeline result in token_headroom mode
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    # Keep original_tokens as the REAL original (pre-Zone-1-swap)
                    # so tokens_saved captures both Zone 1 + Zone 2 savings.
                    optimized_tokens = result.tokens_after
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.openai_pipeline.apply(
                                messages=messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(messages),
                                frozen_message_count=openai_frozen_count,
                                biases=_hook_biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != messages:
                        optimized_messages = result.messages
                        transforms_applied = result.transforms_applied
                        pipeline_timing = result.timing
                        original_tokens = result.tokens_before
                        optimized_tokens = result.tokens_after

                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")
                # Flag compression failure for observability
                _compression_failed = True

        tokens_saved = max(0, original_tokens - optimized_tokens)
        optimization_latency = (time.time() - start_time) * 1000

        # Hook: post_compress
        if self.config.hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            try:
                self.config.hooks.post_compress(
                    CompressEvent(
                        tokens_before=original_tokens,
                        tokens_after=optimized_tokens,
                        tokens_saved=tokens_saved,
                        compression_ratio=tokens_saved / original_tokens
                        if original_tokens > 0
                        else 0,
                        transforms_applied=transforms_applied,
                        model=model,
                        provider="openai",
                    )
                )
            except Exception as e:
                logger.debug(f"[{request_id}] post_compress hook error: {e}")

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        tools = body.get("tools")
        _original_tools = tools  # Preserve for diagnostic / future retry
        if (
            self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions
        ) and not _bypass:
            injector = CCRToolInjector(
                provider="openai",
                inject_tool=self.config.ccr_inject_tool,
                inject_system_instructions=self.config.ccr_inject_system_instructions,
            )
            optimized_messages, tools, was_injected = injector.process_request(
                optimized_messages, tools
            )

            if injector.has_compressed_content:
                if was_injected:
                    logger.debug(
                        f"[{request_id}] CCR: Injected retrieval tool for hashes: {injector.detected_hashes}"
                    )
                else:
                    logger.debug(
                        f"[{request_id}] CCR: Tool already present (MCP?), skipped injection for hashes: {injector.detected_hashes}"
                    )

        # Query Echo: disabled — hurts prefix caching in long conversations.

        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools

        # Route through LiteLLM/any-llm backend if configured
        if self.anthropic_backend is not None:
            try:
                if stream:
                    # Streaming: use stream_openai_message() → SSE events
                    return await self._stream_openai_via_backend(
                        body,
                        headers,
                        model,
                        request_id,
                        start_time,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )
                else:
                    # Non-streaming: use send_openai_message() → JSON
                    backend_response = await self.anthropic_backend.send_openai_message(
                        body, headers
                    )

                    if backend_response.error:
                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=backend_response.body,
                        )

                    # Track metrics
                    total_latency = (time.time() - start_time) * 1000
                    usage = backend_response.body.get("usage", {})
                    output_tokens = usage.get("completion_tokens", 0)
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)

                    await self.metrics.record_request(
                        provider=self.anthropic_backend.name,
                        model=model,
                        input_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        latency_ms=total_latency,
                        cached=False,
                        overhead_ms=optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )

                    if tokens_saved > 0:
                        logger.info(
                            f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                            f"(saved {tokens_saved:,} tokens) via {self.anthropic_backend.name}"
                        )

                    return JSONResponse(
                        status_code=backend_response.status_code,
                        content=backend_response.body,
                    )
            except Exception as e:
                logger.error(f"[{request_id}] Backend error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": str(e),
                            "type": "api_error",
                            "code": "backend_error",
                        }
                    },
                )

        # Direct OpenAI API (no backend configured)
        url = f"{self.OPENAI_API_URL}/v1/chat/completions"

        try:
            if stream:
                # Inject stream_options to get usage stats in streaming response
                # This allows accurate token counting instead of byte-based estimation
                if "stream_options" not in body:
                    body["stream_options"] = {"include_usage": True}
                elif isinstance(body.get("stream_options"), dict):
                    body["stream_options"]["include_usage"] = True

                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=openai_prefix_tracker,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)

                # Full diagnostic dump on upstream errors (OpenAI handler)
                if response.status_code >= 400:
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", {}).get("message", "")
                        err_type = err_body.get("error", {}).get("type", "")
                    except Exception:
                        err_body = {"raw": response.text[:2000]}
                        err_msg = str(response.text[:500])
                        err_type = "parse_error"

                    logger.warning(
                        f"[{request_id}] UPSTREAM_ERROR "
                        f"status={response.status_code} "
                        f"error_type={err_type} "
                        f"error_msg={err_msg!r} "
                        f"model={model} "
                        f"compressed={'yes' if transforms_applied else 'no'} "
                        f"transforms={transforms_applied} "
                        f"original_tokens={original_tokens} "
                        f"optimized_tokens={optimized_tokens} "
                        f"message_count={len(body.get('messages', []))} "
                        f"stream={stream}"
                    )

                    try:
                        debug_dir = Path.home() / ".headroom" / "logs" / "debug_400"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = debug_dir / f"{ts}_{request_id}.json"

                        safe_headers = {}
                        for k, v in headers.items():
                            if k.lower() in ("x-api-key", "authorization"):
                                safe_headers[k] = v[:12] + "..." if v else ""
                            else:
                                safe_headers[k] = v

                        debug_payload = {
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "status_code": response.status_code,
                            "error_response": err_body,
                            "model": model,
                            "stream": stream,
                            "headers": safe_headers,
                            "compression": {
                                "was_compressed": bool(transforms_applied),
                                "transforms": transforms_applied,
                                "original_tokens": original_tokens,
                                "optimized_tokens": optimized_tokens,
                                "tokens_saved": tokens_saved,
                                "compression_failed": _compression_failed,
                            },
                            "tools_sent": body.get("tools"),
                            "tool_count": len(body.get("tools") or []),
                            "original_tool_count": len(_original_tools or []),
                            "messages_sent": body.get("messages"),
                            "message_count": len(body.get("messages", [])),
                            "original_messages": (
                                original_messages
                                if original_messages is not body.get("messages")
                                else "__same_as_sent__"
                            ),
                            "original_message_count": len(original_messages),
                            "system_prompt": body.get("system"),
                        }

                        with open(debug_file, "w") as f:
                            json.dump(debug_payload, f, indent=2, default=str)

                        logger.warning(f"[{request_id}] Full debug dump: {debug_file}")
                    except Exception as dump_err:
                        logger.error(f"[{request_id}] Failed to write debug dump: {dump_err}")

                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    # OpenAI returns cached_tokens in prompt_tokens_details
                    # These are charged at 50% of the input price
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cache_read_tokens = prompt_details.get("cached_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI response: {e}"
                    )

                # Update prefix cache tracker for next turn
                openai_prefix_tracker.update_from_response(
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=0,  # OpenAI doesn't report write tokens
                    messages=optimized_messages,
                )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )

                # Cache
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages, model, response.content, dict(response.headers), tokens_saved
                    )

                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cache_read_tokens,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )

                # Remove compression headers since httpx already decompressed the response
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)  # Length changed after decompression

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(transforms_applied)
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed()
            # Log full error details internally for debugging
            logger.error(f"[{request_id}] OpenAI request failed: {type(e).__name__}: {e}")
            # Return sanitized error message to client (don't expose internal details)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_openai_responses(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/responses endpoint (new Responses API).

        The Responses API differs from /v1/chat/completions:
        - Input: `input` (string or array) instead of `messages`
        - System: `instructions` instead of system message
        - Output: `output[]` array instead of `choices[].message`
        - State: `previous_response_id` for multi-turn
        - Built-in tools: web_search, file_search, code_interpreter
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import (
            COMPRESSION_TIMEOUT_SECONDS,
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.proxy.responses_converter import (
            messages_to_responses_items,
            responses_items_to_messages,
        )
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        model = body.get("model", "unknown")
        stream = body.get("stream", False)

        # Convert Responses API input to messages format for optimization.
        # The Responses API uses a different item model (function_call,
        # function_call_output, reasoning as top-level items) — we convert to
        # Chat Completions messages for the pipeline, then convert back.

        input_data = body.get("input", "")
        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")

        messages: list[dict[str, Any]] = []
        original_items: list[dict[str, Any]] | None = None
        preserved_indices: list[int] = []

        if instructions:
            messages.append({"role": "system", "content": instructions})

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            original_items = input_data
            converted, preserved_indices = responses_items_to_messages(input_data)
            messages.extend(converted)

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Token counting on converted messages
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Optimize: convert items → compress → convert back
        tokens_saved = 0
        transforms_applied: list[str] = []
        optimized_messages = messages
        optimized_tokens = original_tokens

        _bypass = (
            request.headers.get("x-headroom-bypass", "").lower() == "true"
            or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
        )
        _should_compress = (
            self.config.optimize
            and original_items is not None
            and not previous_response_id
            and not _bypass
            and len(messages) > 1
        )
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True

        if _should_compress and _license_ok:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(messages),
                        )
                    ),
                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
            except Exception as e:
                logger.warning(f"[{request_id}] Responses API optimization failed: {e}")

        tokens_saved = max(0, original_tokens - optimized_tokens)
        optimization_latency = (time.time() - start_time) * 1000

        # Convert compressed messages back to Responses API items
        if optimized_messages is not messages and original_items is not None:
            opt_msgs = optimized_messages
            # Strip system message (instructions) — it's separate in Responses API
            if instructions and opt_msgs and opt_msgs[0].get("role") == "system":
                body["instructions"] = opt_msgs[0]["content"]
                opt_msgs = opt_msgs[1:]

            body["input"] = messages_to_responses_items(opt_msgs, original_items, preserved_indices)

        # /v1/responses is OpenAI-specific (Codex) — always routes direct.
        # LiteLLM/AnyLLM backends use /v1/chat/completions or /v1/messages.
        if self.anthropic_backend is not None:
            logger.debug(
                f"[{request_id}] /v1/responses always routes to OpenAI direct "
                f"(backend '{self.anthropic_backend.name}' not used for Responses API)"
            )

        url = f"{self.OPENAI_API_URL}/v1/responses"

        try:
            if stream:
                # Streaming for Responses API uses semantic events
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = original_tokens  # fallback
                output_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", original_tokens)
                    output_tokens = usage.get("output_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI passthrough response: {e}"
                    )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(model, tokens_saved, total_input_tokens)

                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                )

                logger.info(f"[{request_id}] /v1/responses {model}: {total_input_tokens:,} tokens")

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed()
            logger.error(f"[{request_id}] OpenAI responses request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_openai_responses_ws(self, websocket: WebSocket) -> None:
        """WebSocket proxy for /v1/responses (Codex gpt-5.4+).

        Newer Codex versions use WebSocket instead of HTTP POST for the
        Responses API.  This handler:
        1. Accepts the client WebSocket
        2. Receives the first message (``response.create`` request)
        3. Compresses the ``input`` array using the existing pipeline
        4. Opens an upstream WebSocket to OpenAI
        5. Sends the compressed request upstream
        6. Relays all subsequent messages bidirectionally
        """
        from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        try:
            import websockets
        except ImportError:
            await websocket.accept()
            await websocket.close(
                code=1011,
                reason="websockets package not installed. pip install websockets",
            )
            return

        request_id = await self._next_request_id()

        # Forward client headers to upstream, adding required OpenAI-Beta header
        ws_headers = dict(websocket.headers)

        # Extract subprotocol from client — this is an application-level negotiation
        # that MUST be forwarded end-to-end (unlike sec-websocket-key which is per-connection).
        # Codex and OpenAI negotiate a subprotocol; stripping it causes OpenAI to return 500.
        client_subprotocols: list[str] = []
        raw_protocol = ws_headers.get("sec-websocket-protocol", "")
        if raw_protocol:
            client_subprotocols = [p.strip() for p in raw_protocol.split(",") if p.strip()]

        # Accept client connection with the requested subprotocol
        if client_subprotocols:
            await websocket.accept(subprotocol=client_subprotocols[0])
        else:
            await websocket.accept()

        # Build upstream WebSocket URL (http→ws, https→wss)
        base = self.OPENAI_API_URL
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        upstream_url = f"{ws_base}/v1/responses"

        # Forward all client headers except hop-by-hop / per-connection headers.
        # These are WebSocket handshake mechanics that the `websockets` library
        # generates fresh for the upstream connection — forwarding them would conflict.
        # Everything else (auth, org, beta, user-agent, custom headers) is forwarded as-is.
        _skip_headers = frozenset(
            {
                "host",  # must match upstream, not local proxy
                "connection",  # hop-by-hop
                "upgrade",  # hop-by-hop
                "sec-websocket-key",  # per-connection cryptographic nonce
                "sec-websocket-version",  # protocol version (websockets lib sets this)
                "sec-websocket-extensions",  # per-connection negotiation
                "sec-websocket-accept",  # server-side only
                "sec-websocket-protocol",  # handled via subprotocols param below
                "content-length",  # hop-by-hop
                "transfer-encoding",  # hop-by-hop
            }
        )
        upstream_headers: dict[str, str] = {}
        for k, v in ws_headers.items():
            if k.lower() not in _skip_headers:
                upstream_headers[k] = v

        # Ensure Authorization header is present — fall back to OPENAI_API_KEY env var.
        # Safety net for clients that don't forward auth headers via WebSocket upgrade.
        _has_auth = "authorization" in {k.lower() for k in upstream_headers}
        if not _has_auth:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                upstream_headers["Authorization"] = f"Bearer {api_key}"
                logger.debug(f"[{request_id}] WS: injected Authorization from OPENAI_API_KEY env")
            else:
                logger.warning(
                    f"[{request_id}] WS: no Authorization header from client and "
                    f"OPENAI_API_KEY not set — upstream will likely reject"
                )

        # Ensure the required beta header is present — OpenAI returns 500 without it
        if "openai-beta" not in {k.lower() for k in upstream_headers}:
            upstream_headers["OpenAI-Beta"] = "responses-api=v1"

        logger.debug(
            f"[{request_id}] WS upstream headers: "
            f"{[k for k in upstream_headers if k.lower() != 'authorization']}, "
            f"subprotocols={client_subprotocols}"
        )

        try:
            # Receive the first message from client (the response.create request)
            first_msg_raw = await websocket.receive_text()

            # --- Optional: compress the input in the first message ---
            try:
                body = json.loads(first_msg_raw)
                input_data = body.get("input")
                tokens_saved = 0

                should_compress = (
                    self.config.optimize
                    and isinstance(input_data, list)
                    and len(input_data) > 1
                    and not body.get("previous_response_id")
                )
                if should_compress:
                    try:
                        from headroom.proxy.responses_converter import (
                            messages_to_responses_items,
                            responses_items_to_messages,
                        )

                        model = body.get("model", "gpt-4o")
                        converted, preserved = responses_items_to_messages(input_data)

                        messages: list[dict[str, Any]] = []
                        instructions = body.get("instructions")
                        if instructions:
                            messages.append({"role": "system", "content": instructions})
                        messages.extend(converted)

                        tokenizer = get_tokenizer(model)
                        original_tokens = tokenizer.count_messages(messages)

                        context_limit = self.openai_provider.get_context_limit(model)
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                lambda: self.openai_pipeline.apply(
                                    messages=messages,
                                    model=model,
                                    model_limit=context_limit,
                                    context=extract_user_query(messages),
                                )
                            ),
                            timeout=COMPRESSION_TIMEOUT_SECONDS,
                        )

                        if result.messages != messages:
                            opt = result.messages
                            if instructions and opt and opt[0].get("role") == "system":
                                body["instructions"] = opt[0]["content"]
                                opt = opt[1:]
                            body["input"] = messages_to_responses_items(opt, input_data, preserved)
                            tokens_saved = max(0, original_tokens - result.tokens_after)
                            first_msg_raw = json.dumps(body)
                            logger.info(
                                f"[{request_id}] WS /v1/responses compressed: "
                                f"saved {tokens_saved} tokens"
                            )
                    except Exception as e:
                        logger.warning(f"[{request_id}] WS compression failed: {e}")

            except json.JSONDecodeError:
                # Not JSON — pass through as-is
                tokens_saved = 0

            # --- Connect to upstream OpenAI WebSocket ---
            logger.debug(f"[{request_id}] WS connecting to {upstream_url}")

            # Use ssl=True to let the websockets library handle SSL natively.
            # Manual ssl.create_default_context() + certifi doesn't load the
            # Windows system cert store, causing HTTP 500 on wss:// connections.
            use_ssl: bool | None = True if upstream_url.startswith("wss://") else None

            async with websockets.connect(
                upstream_url,
                additional_headers=upstream_headers,
                subprotocols=(
                    [websockets.Subprotocol(p) for p in client_subprotocols]
                    if client_subprotocols and hasattr(websockets, "Subprotocol")
                    else client_subprotocols or None
                ),
                ssl=use_ssl,
            ) as upstream:
                # Send (potentially compressed) first message
                await upstream.send(first_msg_raw)

                # Bidirectional relay
                async def _client_to_upstream() -> None:
                    try:
                        while True:
                            msg = await websocket.receive_text()
                            await upstream.send(msg)
                    except Exception as relay_err:
                        if "WebSocketDisconnect" not in type(relay_err).__name__:
                            logger.debug(
                                f"[{request_id}] WS client→upstream relay ended: {relay_err}"
                            )
                        with contextlib.suppress(Exception):
                            await upstream.close()

                async def _upstream_to_client() -> None:
                    try:
                        async for msg in upstream:
                            if isinstance(msg, str):
                                await websocket.send_text(msg)
                            elif isinstance(msg, bytes):
                                await websocket.send_bytes(msg)
                            else:
                                await websocket.send_text(str(msg))
                    except Exception as relay_err:
                        if "WebSocketDisconnect" not in type(relay_err).__name__:
                            logger.debug(
                                f"[{request_id}] WS upstream→client relay ended: {relay_err}"
                            )
                    finally:
                        with contextlib.suppress(Exception):
                            await websocket.close()

                await asyncio.gather(
                    _client_to_upstream(),
                    _upstream_to_client(),
                    return_exceptions=True,
                )

            # Record metrics
            if tokens_saved > 0:
                model_name = body.get("model", "unknown") if isinstance(body, dict) else "unknown"
                await self.metrics.record_request(
                    provider="openai",
                    model=model_name,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=tokens_saved,
                    latency_ms=0,
                )

        except Exception as e:
            if "WebSocketDisconnect" not in type(e).__name__:
                # Extract response body from websockets InvalidStatus for better debugging
                error_detail = str(e)
                if hasattr(e, "response"):
                    try:
                        resp = e.response
                        body_bytes = getattr(resp, "body", None) or b""
                        if body_bytes:
                            error_detail += (
                                f" | body: {body_bytes[:500].decode('utf-8', errors='replace')}"
                            )
                    except Exception:
                        pass
                logger.error(f"[{request_id}] WS proxy error: {error_detail}")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason=str(e)[:120])

    async def handle_compress(self, request: Request) -> JSONResponse:
        """Compress messages without calling an LLM.

        POST /v1/compress
        Body: {"messages": [...], "model": "...", "config": {}}
        Returns compressed messages + metrics.
        """
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json

        # Check bypass header
        if request.headers.get("x-headroom-bypass", "").lower() == "true":
            try:
                body = await _read_request_json(request)
            except (json.JSONDecodeError, ValueError) as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid request body: {e!s}"},
                )
            messages = body.get("messages", [])
            return JSONResponse(
                {
                    "messages": messages,
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            body = await _read_request_json(request)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Invalid JSON in request body.",
                    }
                },
            )

        messages = body.get("messages")
        model = body.get("model")

        if messages is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: messages",
                    }
                },
            )

        if model is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: model",
                    }
                },
            )

        if not messages:
            return JSONResponse(
                {
                    "messages": [],
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            # Use OpenAI pipeline (messages are in OpenAI format from TS SDK)
            # Allow optional token_budget to override model's context limit
            # (used by OpenClaw compact() and other callers that need tighter budgets)
            token_budget = body.get("token_budget")
            context_limit = (
                token_budget
                if token_budget and isinstance(token_budget, int)
                else self.openai_provider.get_context_limit(model)
            )
            result = self.openai_pipeline.apply(
                messages=messages,
                model=model,
                model_limit=context_limit,
            )

            return JSONResponse(
                {
                    "messages": result.messages,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "tokens_saved": result.tokens_before - result.tokens_after,
                    "compression_ratio": (
                        result.tokens_after / result.tokens_before
                        if result.tokens_before > 0
                        else 1.0
                    ),
                    "transforms_applied": result.transforms_applied,
                    "transforms_summary": result.transforms_summary,
                    "ccr_hashes": result.markers_inserted,
                }
            )
        except Exception as e:
            logger.exception("Compression failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "compression_error",
                        "message": str(e),
                    }
                },
            )

    async def handle_passthrough(
        self,
        request: Request,
        base_url: str,
        endpoint_name: str | None = None,
        provider: str | None = None,
    ) -> Response:
        """Pass through request unchanged.

        Args:
            request: The incoming request
            base_url: The upstream API base URL
            endpoint_name: Optional name for stats tracking (e.g., "models", "embeddings")
            provider: Optional provider name for stats (e.g., "openai", "anthropic", "gemini")
        """
        from fastapi.responses import Response

        start_time = time.time()
        path = request.url.path
        url = f"{base_url}{path}"

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        body = await request.body()

        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        # Remove compression headers since httpx already decompressed the response
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)  # Length changed after decompression

        # Track stats for passthrough requests
        if endpoint_name and provider:
            latency_ms = (time.time() - start_time) * 1000
            await self.metrics.record_request(
                provider=provider,
                model=f"passthrough:{endpoint_name}",
                input_tokens=0,
                output_tokens=0,
                tokens_saved=0,
                latency_ms=latency_ms,
                cached=False,
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_databricks_invocations(
        self,
        request: Request,
        model: str,
    ) -> Response | StreamingResponse:
        """Handle Databricks native /serving-endpoints/{model}/invocations endpoint.

        This enables using the Databricks CLI directly with Headroom:
            databricks serving-endpoints query <model> --profile HEADROOM --json '{"messages": [...]}'

        The request/response format is identical to OpenAI chat completions,
        so we inject the model from the path and delegate to handle_openai_chat.
        """
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json

        request_id = await self._next_request_id()

        try:
            body = await _read_request_json(request)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to parse Databricks request body: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}
                },
            )

        # Inject model from path into body (Databricks CLI passes model in URL, not body)
        body["model"] = model

        logger.info(f"[{request_id}] Databricks invocation: model={model}")

        # Create a new request with the modified body
        # We reuse the OpenAI chat handler since the format is identical
        from starlette.requests import Request as StarletteRequest

        # Build new scope with the body already parsed
        scope = dict(request.scope)

        # Create a simple receive function that returns our modified body
        body_bytes = json.dumps(body).encode()

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        modified_request = StarletteRequest(scope, receive)

        # Delegate to the OpenAI chat handler (same format)
        return await self.handle_openai_chat(modified_request)
