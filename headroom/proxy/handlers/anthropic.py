"""Anthropic handler mixin for HeadroomProxy.

Contains all Anthropic Messages API handlers including batch operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse

import httpx

logger = logging.getLogger("headroom.proxy")


class AnthropicHandlerMixin:
    """Mixin providing Anthropic API handler methods for HeadroomProxy."""

    async def handle_anthropic_messages(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle Anthropic /v1/messages endpoint."""
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.cache.compression_store import get_compression_store
        from headroom.ccr import CCRToolInjector
        from headroom.proxy.cost import _summarize_transforms
        from headroom.proxy.helpers import (
            MAX_MESSAGE_ARRAY_LENGTH,
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.proxy.models import RequestLog
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
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                    },
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid request body: {e!s}",
                    },
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        # Validate message array size
        if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Message array too large ({len(messages)} messages). "
                        f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                    },
                },
            )

        stream = body.get("stream", False)

        # Bypass: skip ALL compression, TOIN learning, and CCR injection
        # when the caller explicitly opts out via header.
        # Prevents Headroom from corrupting sub-agent API calls
        # (e.g., Claude Code sub-agents that inherit ANTHROPIC_BASE_URL).
        _bypass = (
            request.headers.get("x-headroom-bypass", "").lower() == "true"
            or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
        )
        if _bypass:
            logger.info(f"[{request_id}] Bypass: skipping compression (header)")

        # TODO: Re-enable image compression once token counting is accurate.
        # Image compression was disabled because the tokenizer counted base64
        # image data as text tokens (330K per 1MB image), inflating savings by
        # 3-4x. The compressor itself works, but reported metrics were wrong.
        # To re-enable: fix tokenizer to extract image dimensions and use
        # Anthropic's formula (width*height/750), then uncomment below.
        #
        # if self.config.image_optimize and messages and not _bypass:
        #     compressor = _get_image_compressor()
        #     if compressor and compressor.has_images(messages):
        #         messages = compressor.compress(messages, provider="anthropic")
        #         if compressor.last_result:
        #             logger.info(
        #                 f"Image compression: {compressor.last_result.technique.value} "
        #                 f"({compressor.last_result.savings_percent:.0f}% saved, "
        #                 f"{compressor.last_result.original_tokens} -> "
        #                 f"{compressor.last_result.compressed_tokens} tokens)"
        #             )

        # Extract headers and tags
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)

        # Rate limiting
        if self.rate_limiter:
            api_key = headers.get("x-api-key", "")
            client_ip = request.client.host if request.client else "unknown"
            rate_key = f"{api_key[:16]}:{client_ip}" if api_key else client_ip
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                    headers={"Retry-After": str(int(wait_seconds) + 1)},
                )

        # Budget check
        if self.cost_tracker:
            allowed, remaining = self.cost_tracker.check_budget()
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Budget exceeded for {self.config.budget_period} period",
                )

        # Memory: Get user ID when memory is enabled (fallback to "default" for simple DevEx)
        memory_user_id: str | None = None
        if self.memory_handler:
            memory_user_id = headers.get("x-headroom-user-id", "default")

        # Check cache (non-streaming only)
        cache_hit = False
        if self.cache and not stream:
            cached = await self.cache.get(messages, model)
            if cached:
                cache_hit = True
                optimization_latency = (time.time() - start_time) * 1000

                await self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=0,  # Savings already counted when response was cached
                    latency_ms=optimization_latency,
                    cached=True,
                )

                # Remove compression headers from cached response
                response_headers = dict(cached.response_headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=cached.response_body,
                    headers=response_headers,
                    media_type="application/json",
                )

        # Count original tokens
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Hook: pre_compress — let hooks modify messages before compression

        if self.config.hooks:
            from headroom.hooks import CompressContext

            _hook_ctx = CompressContext(
                model=model,
                user_query=extract_user_query(messages),
                provider="anthropic",
            )
            try:
                messages = self.config.hooks.pre_compress(messages, _hook_ctx)
            except Exception as e:
                logger.debug(f"[{request_id}] pre_compress hook error: {e}")

        # Apply optimization
        transforms_applied = []
        pipeline_timing: dict[str, float] = {}
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        # Get prefix cache tracker for this session
        session_id = self.session_tracker_store.compute_session_id(request, model, messages)
        prefix_tracker = self.session_tracker_store.get_or_create(session_id, "anthropic")
        frozen_message_count = prefix_tracker.get_frozen_message_count()

        _compression_failed = False
        original_messages = messages  # Preserve for 400-retry fallback
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and not _bypass and _license_ok:
            try:
                from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS

                context_limit = self.anthropic_provider.get_context_limit(model)
                biases = (
                    self.config.hooks.compute_biases(messages, _hook_ctx)
                    if self.config.hooks
                    else None
                )

                if self.config.mode == "token_headroom":
                    comp_cache = self._get_compression_cache(session_id)

                    # Zone 1: Swap cached compressed versions into working copy
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary: consecutive stable messages from start
                    frozen_message_count = comp_cache.compute_frozen_count(messages)

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.anthropic_pipeline.apply(
                                messages=working_messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(working_messages),
                                frozen_message_count=frozen_message_count,
                                biases=biases,
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    # Cache newly compressed messages (index-aligned diff)
                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)

                    # Always use pipeline result — Zone 1 swaps are already applied
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    # Keep original_tokens as the REAL original (pre-Zone-1-swap)
                    # so tokens_saved captures both Zone 1 + Zone 2 savings.
                    # original_tokens was set at line ~2183 from uncompressed messages.
                    optimized_tokens = result.tokens_after
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: self.anthropic_pipeline.apply(
                                messages=messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(messages),
                                frozen_message_count=frozen_message_count,
                                biases=biases,
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

        # Hook: post_compress — let hooks observe compression results
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
                        user_query=_hook_ctx.user_query if self.config.hooks else "",
                        provider="anthropic",
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
            # Create fresh injector to avoid state leakage between requests
            injector = CCRToolInjector(
                provider="anthropic",
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

                # Track compression in context tracker for multi-turn awareness
                if self.ccr_context_tracker:
                    self._turn_counter += 1
                    for hash_key in injector.detected_hashes:
                        # Get compression metadata from store
                        store = get_compression_store()
                        entry = store.get_metadata(hash_key)
                        if entry:
                            self.ccr_context_tracker.track_compression(
                                hash_key=hash_key,
                                turn_number=self._turn_counter,
                                tool_name=entry.get("tool_name"),
                                original_count=entry.get("original_item_count", 0),
                                compressed_count=entry.get("compressed_item_count", 0),
                                query_context=entry.get("query_context", ""),
                                sample_content=entry.get("compressed_content", "")[:500],
                            )

        # CCR Proactive Expansion: Check if current query needs expanded context
        if self.ccr_context_tracker and self.config.ccr_proactive_expansion:
            # Extract user query from messages
            user_query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_query = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                user_query = block.get("text", "")
                                break
                    break

            if user_query:
                recommendations = self.ccr_context_tracker.analyze_query(
                    user_query, self._turn_counter
                )
                if recommendations:
                    expansions = self.ccr_context_tracker.execute_expansions(recommendations)
                    if expansions:
                        # Add expanded context to the system message or as additional context
                        expansion_text = self.ccr_context_tracker.format_expansions_for_context(
                            expansions
                        )
                        logger.info(
                            f"[{request_id}] CCR: Proactively expanded {len(expansions)} context(s) "
                            f"based on query relevance"
                        )
                        # Append to the last user message
                        if optimized_messages and optimized_messages[-1].get("role") == "user":
                            last_msg = optimized_messages[-1]
                            content = last_msg.get("content", "")
                            if isinstance(content, str):
                                optimized_messages[-1] = {
                                    **last_msg,
                                    "content": content + "\n\n" + expansion_text,
                                }

        # Traffic Learner: Extract patterns from inbound tool results
        if self.traffic_learner:
            try:
                # Wire backend on first use (lazy init after memory handler is ready)
                if (
                    self.traffic_learner._backend is None
                    and self.memory_handler
                    and self.memory_handler.initialized
                    and self.memory_handler.backend
                ):
                    self.traffic_learner.set_backend(self.memory_handler.backend)

                # Extract tool results from messages and learn from them
                tool_results = self.traffic_learner.extract_tool_results_from_messages(
                    optimized_messages
                )
                for tr in tool_results[-5:]:  # Only recent results
                    await self.traffic_learner.on_tool_result(
                        tool_name=tr["tool_name"],
                        tool_input=tr["input"],
                        tool_output=tr["output"],
                        is_error=tr["is_error"],
                    )

                # Also extract preference signals from user messages
                await self.traffic_learner.on_messages(optimized_messages)
            except Exception as e:
                logger.debug(f"[{request_id}] Traffic learner: {e}")

        # Memory: Inject context and tools
        if self.memory_handler and memory_user_id:
            # Search and inject memory context
            if self.memory_handler.config.inject_context:
                try:
                    memory_context = await self.memory_handler.search_and_format_context(
                        memory_user_id, optimized_messages
                    )
                    if memory_context:
                        optimized_messages = self._inject_system_context(
                            optimized_messages, memory_context, body=body
                        )
                        logger.info(
                            f"[{request_id}] Memory: Injected {len(memory_context)} chars of context"
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Memory: Context injection failed: {e}")

            # Inject memory tools
            if self.memory_handler.config.inject_tools:
                tools, mem_tools_injected = self.memory_handler.inject_tools(tools, "anthropic")
                if mem_tools_injected:
                    tool_names = [
                        t.get("name") or t.get("type", "")
                        for t in tools
                        if t.get("name", "").startswith("memory")
                        or t.get("type", "").startswith("memory")
                    ]
                    logger.info(f"[{request_id}] Memory: Injected tools: {tool_names}")

                    # Add beta headers for native memory tool
                    beta_headers = self.memory_handler.get_beta_headers()
                    if beta_headers:
                        for key, value in beta_headers.items():
                            # Merge with existing beta header if present
                            existing = headers.get(key, "")
                            if existing and value not in existing:
                                headers[key] = f"{existing},{value}"
                            else:
                                headers[key] = value
                            logger.info(
                                f"[{request_id}] Memory: Added beta header: {key}={headers[key]}"
                            )

        # Query Echo: disabled — hurts prefix caching in long conversations.
        # The echo changes every turn, invalidating the cached prefix.
        # To re-enable, uncomment and set query_echo_enabled on ProxyConfig.

        # Update body
        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools

        # Forward request - use Bedrock backend if configured, otherwise direct API
        if self.anthropic_backend is not None:
            # Route through Bedrock backend
            try:
                if stream:
                    return await self._stream_response_bedrock(
                        body,
                        headers,
                        "anthropic",
                        model,
                        request_id,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )
                else:
                    backend_response = await self.anthropic_backend.send_message(body, headers)

                    if backend_response.error:
                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=backend_response.body,
                        )

                    # Track metrics
                    total_latency = (time.time() - start_time) * 1000
                    usage = backend_response.body.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                    _backend_name = (
                        self.anthropic_backend.name if self.anthropic_backend else "anthropic"
                    )
                    await self.metrics.record_request(
                        provider=_backend_name,
                        model=model,
                        input_tokens=optimized_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        latency_ms=total_latency,
                        cached=False,
                        overhead_ms=optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )

                    if self.cost_tracker:
                        self.cost_tracker.record_tokens(model, tokens_saved, optimized_tokens)

                    # Log request
                    if self.logger:
                        self.logger.log(
                            RequestLog(
                                request_id=request_id,
                                timestamp=datetime.now().isoformat(),
                                provider=_backend_name,
                                model=model,
                                input_tokens_original=original_tokens,
                                input_tokens_optimized=optimized_tokens,
                                output_tokens=output_tokens,
                                tokens_saved=tokens_saved,
                                savings_percent=(tokens_saved / original_tokens * 100)
                                if original_tokens > 0
                                else 0,
                                optimization_latency_ms=optimization_latency,
                                total_latency_ms=total_latency,
                                tags=tags,
                                cache_hit=False,
                                transforms_applied=transforms_applied,
                                request_messages=body.get("messages")
                                if self.config.log_full_messages
                                else None,
                            )
                        )

                    return JSONResponse(
                        status_code=backend_response.status_code,
                        content=backend_response.body,
                    )
            except Exception as e:
                logger.error(f"[{request_id}] Bedrock backend error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e)},
                    },
                )

        # Direct Anthropic API
        url = f"{self.ANTHROPIC_API_URL}/v1/messages"

        try:
            if stream:
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "anthropic",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    memory_user_id=memory_user_id,
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=prefix_tracker,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)

                # Full diagnostic dump on upstream errors.
                # Writes pre/post compression messages, tools, and error
                # to ~/.headroom/logs/debug_400/ for offline analysis.
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

                    # Dump full request details to debug file
                    try:
                        debug_dir = Path.home() / ".headroom" / "logs" / "debug_400"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = debug_dir / f"{ts}_{request_id}.json"

                        # Sanitize headers (redact API keys)
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

                # Parse response for CCR handling
                resp_json = None
                try:
                    resp_json = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to parse response JSON for CCR handling: {e}"
                    )

                # CCR Response Handling: Handle headroom_retrieve tool calls automatically
                if (
                    self.ccr_response_handler
                    and resp_json
                    and response.status_code == 200
                    and self.ccr_response_handler.has_ccr_tool_calls(resp_json, "anthropic")
                ):
                    logger.info(f"[{request_id}] CCR: Detected retrieval tool call, handling...")

                    # Create API call function for continuation
                    # Use a fresh client to avoid potential decompression state issues
                    async def api_call_fn(
                        msgs: list[dict], tls: list[dict] | None
                    ) -> dict[str, Any]:
                        continuation_body = {
                            **body,
                            "messages": msgs,
                        }
                        if tls is not None:
                            continuation_body["tools"] = tls

                        # Use clean headers for continuation
                        continuation_headers = {
                            k: v
                            for k, v in headers.items()
                            if k.lower()
                            not in (
                                "content-encoding",
                                "transfer-encoding",
                                "accept-encoding",
                                "content-length",
                            )
                        }

                        # Reuse main client for CCR continuations (connection pooling)
                        logger.info(f"CCR: Making continuation request with {len(msgs)} messages")
                        assert self.http_client is not None, "HTTP client not initialized"
                        try:
                            cont_response = await self.http_client.post(
                                url,
                                json=continuation_body,
                                headers=continuation_headers,
                                timeout=httpx.Timeout(120.0),  # Override timeout for CCR
                            )
                            logger.info(
                                f"CCR: Got response status={cont_response.status_code}, "
                                f"content-encoding={cont_response.headers.get('content-encoding')}"
                            )
                            result: dict[str, Any] = cont_response.json()
                            logger.info("CCR: Parsed JSON successfully")
                            return result
                        except Exception as e:
                            resp_headers: str | dict[str, str] = "N/A"
                            try:
                                resp_headers = dict(cont_response.headers)
                            except Exception:
                                pass
                            logger.error(
                                f"CCR: API call failed: {e}, response headers: {resp_headers}"
                            )
                            raise

                    # Handle CCR tool calls
                    try:
                        final_resp_json = await self.ccr_response_handler.handle_response(
                            resp_json,
                            optimized_messages,
                            tools,
                            api_call_fn,
                            provider="anthropic",
                        )
                        # Update response content with final response
                        resp_json = final_resp_json
                        # Remove encoding headers since content is now uncompressed JSON
                        ccr_response_headers = {
                            k: v
                            for k, v in response.headers.items()
                            if k.lower() not in ("content-encoding", "content-length")
                        }
                        try:
                            ccr_content = json.dumps(final_resp_json).encode()
                        except (TypeError, ValueError) as json_err:
                            logger.warning(
                                f"[{request_id}] CCR: JSON serialization failed: {json_err}"
                            )
                            ccr_content = json.dumps(resp_json).encode()
                        response = httpx.Response(
                            status_code=200,
                            content=ccr_content,
                            headers=ccr_response_headers,
                        )
                        logger.info(f"[{request_id}] CCR: Retrieval handled successfully")
                    except Exception as e:
                        import traceback

                        logger.warning(
                            f"[{request_id}] CCR: Response handling failed: {e}\n"
                            f"Traceback: {traceback.format_exc()}"
                        )
                        # Continue with original response

                # Memory: Handle memory tool calls in response
                if (
                    self.memory_handler
                    and memory_user_id
                    and resp_json
                    and response.status_code == 200
                    and self.memory_handler.has_memory_tool_calls(resp_json, "anthropic")
                ):
                    logger.info(f"[{request_id}] Memory: Detected memory tool call, handling...")

                    try:
                        # Execute memory tool calls
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            resp_json, memory_user_id, "anthropic"
                        )

                        if tool_results:
                            # Create continuation messages
                            assistant_msg = {
                                "role": "assistant",
                                "content": resp_json.get("content", []),
                            }
                            user_msg = {
                                "role": "user",
                                "content": tool_results,
                            }

                            continuation_messages = optimized_messages + [assistant_msg, user_msg]

                            # Make continuation API call
                            continuation_body = {**body, "messages": continuation_messages}
                            if tools:
                                continuation_body["tools"] = tools

                            cont_response = await self._retry_request(
                                "POST", url, headers, continuation_body
                            )

                            # Update response with continuation
                            resp_json = cont_response.json()
                            response = cont_response
                            logger.info(
                                f"[{request_id}] Memory: Tool calls handled, continuation complete"
                            )

                    except Exception as e:
                        logger.warning(f"[{request_id}] Memory: Tool call handling failed: {e}")
                        # Continue with original response

                total_latency = (time.time() - start_time) * 1000

                # Parse response for output token count and cache metrics
                output_tokens = 0
                cr_tokens = 0
                cw_tokens = 0
                uncached_input_tokens = 0
                if resp_json:
                    usage = resp_json.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)
                    cr_tokens = usage.get("cache_read_input_tokens", 0)
                    cw_tokens = usage.get("cache_creation_input_tokens", 0)
                    uncached_input_tokens = usage.get("input_tokens", 0)

                # Update prefix cache tracker for next turn
                prefix_tracker.update_from_response(
                    cache_read_tokens=cr_tokens,
                    cache_write_tokens=cw_tokens,
                    messages=optimized_messages,
                )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cr_tokens,
                        cache_write_tokens=cw_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )

                # Cache response
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved=tokens_saved,
                    )

                # Record metrics — use optimized_tokens (what we sent), not API's
                # input_tokens which is just the non-cached portion with prompt caching
                await self.metrics.record_request(
                    provider="anthropic",
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cr_tokens,
                    cache_write_tokens=cw_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                )

                # Log request
                if self.logger:
                    self.logger.log(
                        RequestLog(
                            request_id=request_id,
                            timestamp=datetime.now().isoformat(),
                            provider="anthropic",
                            model=model,
                            input_tokens_original=original_tokens,
                            input_tokens_optimized=optimized_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            savings_percent=(tokens_saved / original_tokens * 100)
                            if original_tokens > 0
                            else 0,
                            optimization_latency_ms=optimization_latency,
                            total_latency_ms=total_latency,
                            tags=tags,
                            cache_hit=cache_hit,
                            transforms_applied=transforms_applied,
                            waste_signals=waste_signals_dict,
                            request_messages=messages if self.config.log_full_messages else None,
                        )
                    )

                # Structured perf log line for `headroom perf` analysis
                num_msgs = len(messages)
                resp_usage = resp_json.get("usage", {}) if resp_json else {}
                cr = resp_usage.get("cache_read_input_tokens", 0)
                cw = resp_usage.get("cache_creation_input_tokens", 0)
                chp = round(cr / (cr + cw) * 100) if (cr + cw) > 0 else 0
                timing_str = (
                    " ".join(f"{k}={v:.0f}ms" for k, v in pipeline_timing.items())
                    if pipeline_timing
                    else ""
                )
                logger.info(
                    f"[{request_id}] PERF "
                    f"model={model} msgs={num_msgs} "
                    f"tok_before={original_tokens} tok_after={optimized_tokens} "
                    f"tok_saved={tokens_saved} "
                    f"cache_read={cr} cache_write={cw} cache_hit_pct={chp} "
                    f"opt_ms={optimization_latency:.0f} "
                    f"transforms={_summarize_transforms(transforms_applied)}"
                    f"{' timing=' + timing_str if timing_str else ''}"
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
                if cache_hit:
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
            logger.error(f"[{request_id}] Request failed: {type(e).__name__}: {e}")

            # Try fallback if enabled
            if self.config.fallback_enabled and self.config.fallback_provider == "openai":
                logger.info(f"[{request_id}] Attempting fallback to OpenAI")
                # Convert to OpenAI format and retry
                # (simplified - would need message format conversion)

            # Return sanitized error message to client (don't expose internal details)
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "An error occurred while processing your request. Please try again.",
                    },
                },
            )

    async def handle_anthropic_batch_create(
        self,
        request: Request,
    ) -> Response:
        """Handle Anthropic POST /v1/messages/batches endpoint with compression.

        Anthropic batch format:
        {
            "requests": [
                {
                    "custom_id": "req-1",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                },
                ...
            ]
        }

        This method applies compression to each request's messages before forwarding.
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import CCRToolInjector
        from headroom.proxy.helpers import MAX_REQUEST_BODY_SIZE, _read_request_json
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                    },
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid request body: {e!s}",
                    },
                },
            )

        requests_list = body.get("requests", [])
        if not requests_list:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Missing or empty 'requests' field in batch request",
                    },
                },
            )

        # Extract headers
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Track compression stats across all batch requests
        total_original_tokens = 0
        total_optimized_tokens = 0
        total_tokens_saved = 0
        compressed_requests = []
        pipeline_timing: dict[str, float] = {}

        # Apply compression to each request in the batch
        for batch_req in requests_list:
            custom_id = batch_req.get("custom_id", "")
            params = batch_req.get("params", {})
            messages = params.get("messages", [])
            model = params.get("model", "unknown")

            if not messages or not self.config.optimize:
                # No messages or optimization disabled - pass through unchanged
                compressed_requests.append(batch_req)
                continue

            # Apply optimization
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                result = self.anthropic_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )

                optimized_messages = result.messages
                for k, v in result.timing.items():
                    pipeline_timing[k] = pipeline_timing.get(k, 0.0) + v
                # Use pipeline's token counts for consistency with pipeline logs
                original_tokens = result.tokens_before
                optimized_tokens = result.tokens_after
                total_original_tokens += original_tokens
                total_optimized_tokens += optimized_tokens
                tokens_saved = max(0, original_tokens - optimized_tokens)
                total_tokens_saved += tokens_saved

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = params.get("tools")
                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="anthropic",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    optimized_messages, tools, was_injected = injector.process_request(
                        optimized_messages, tools
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for batch request '{custom_id}'"
                        )

                # Create compressed batch request
                compressed_params = {**params, "messages": optimized_messages}
                if tools is not None:
                    compressed_params["tools"] = tools
                compressed_requests.append(
                    {
                        "custom_id": custom_id,
                        "params": compressed_params,
                    }
                )

                if tokens_saved > 0:
                    logger.debug(
                        f"[{request_id}] Batch request '{custom_id}': "
                        f"{original_tokens:,} -> {optimized_tokens:,} tokens "
                        f"(saved {tokens_saved:,})"
                    )

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Optimization failed for batch request '{custom_id}': {e}"
                )
                # Pass through unchanged on failure
                compressed_requests.append(batch_req)
                total_optimized_tokens += original_tokens

        # Update body with compressed requests
        body["requests"] = compressed_requests

        optimization_latency = (time.time() - start_time) * 1000

        # Forward request to Anthropic
        url = f"{self.ANTHROPIC_API_URL}/v1/messages/batches"

        try:
            response = await self._retry_request("POST", url, headers, body)

            # Record metrics
            await self.metrics.record_request(
                provider="anthropic",
                model="batch",
                input_tokens=total_optimized_tokens,
                output_tokens=0,
                tokens_saved=total_tokens_saved,
                latency_ms=optimization_latency,
                overhead_ms=optimization_latency,
                pipeline_timing=pipeline_timing,
            )

            # Log compression stats
            if total_tokens_saved > 0:
                savings_percent = (
                    (total_tokens_saved / total_original_tokens * 100)
                    if total_original_tokens > 0
                    else 0
                )
                logger.info(
                    f"[{request_id}] Batch ({len(compressed_requests)} requests): "
                    f"{total_original_tokens:,} -> {total_optimized_tokens:,} tokens "
                    f"(saved {total_tokens_saved:,}, {savings_percent:.1f}%)"
                )

            # Store batch context for CCR result processing
            if response.status_code == 200 and self.config.ccr_inject_tool:
                try:
                    response_data = response.json()
                    batch_id = response_data.get("id")
                    if batch_id:
                        await self._store_anthropic_batch_context(
                            batch_id,
                            requests_list,
                            headers.get("x-api-key"),
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to store batch context: {e}")

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
            logger.error(f"[{request_id}] Batch request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "An error occurred while processing your batch request. Please try again.",
                    },
                },
            )

    async def handle_anthropic_batch_passthrough(
        self,
        request: Request,
        batch_id: str | None = None,
    ) -> Response:
        """Handle Anthropic batch passthrough endpoints.

        Used for:
        - GET /v1/messages/batches - List batches
        - GET /v1/messages/batches/{batch_id} - Get batch
        - GET /v1/messages/batches/{batch_id}/results - Get batch results
        - POST /v1/messages/batches/{batch_id}/cancel - Cancel batch
        """
        from fastapi.responses import Response

        start_time = time.time()
        path = request.url.path
        url = f"{self.ANTHROPIC_API_URL}{path}"

        # Preserve query string parameters (e.g., limit, after_id for list endpoint)
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

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="anthropic",
            model="passthrough:batches",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        # Remove compression headers
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _store_anthropic_batch_context(
        self,
        batch_id: str,
        requests_list: list[dict[str, Any]],
        api_key: str | None,
    ) -> None:
        """Store batch context for CCR result processing.

        Args:
            batch_id: The batch ID from the API response.
            requests_list: The original batch requests.
            api_key: The API key for continuation calls.
        """
        from headroom.ccr import BatchContext, BatchRequestContext, get_batch_context_store

        store = get_batch_context_store()
        context = BatchContext(
            batch_id=batch_id,
            provider="anthropic",
            api_key=api_key,
            api_base_url=self.ANTHROPIC_API_URL,
        )

        for batch_req in requests_list:
            custom_id = batch_req.get("custom_id", "")
            params = batch_req.get("params", {})
            context.add_request(
                BatchRequestContext(
                    custom_id=custom_id,
                    messages=params.get("messages", []),
                    tools=params.get("tools"),
                    model=params.get("model", ""),
                    extras={
                        "max_tokens": params.get("max_tokens", 4096),
                        "system": params.get("system"),
                    },
                )
            )

        await store.store(context)
        logger.debug(f"Stored batch context for {batch_id} with {len(requests_list)} requests")

    async def handle_anthropic_batch_results(
        self,
        request: Request,
        batch_id: str,
    ) -> Response:
        """Handle Anthropic batch results with CCR post-processing.

        This endpoint:
        1. Fetches raw results from Anthropic
        2. Detects CCR tool calls in each result
        3. Executes retrieval and makes continuation calls
        4. Returns processed results with complete responses
        """
        from fastapi.responses import Response

        from headroom.ccr import BatchResultProcessor, get_batch_context_store

        start_time = time.time()

        # Forward request to get raw results
        url = f"{self.ANTHROPIC_API_URL}/v1/messages/batches/{batch_id}/results"

        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)

        response = await self.http_client.get(url, headers=headers)  # type: ignore[union-attr]

        if response.status_code != 200:
            # Error - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Parse results - Anthropic batch results are JSONL format
        raw_content = response.content.decode("utf-8")
        results = []
        for line in raw_content.strip().split("\n"):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not results:
            # No results to process
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if we have context and CCR processing is enabled
        store = get_batch_context_store()
        batch_context = await store.get(batch_id)

        if batch_context is None or not self.config.ccr_inject_tool:
            # No context or CCR disabled - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Process results with CCR handler
        processor = BatchResultProcessor(self.http_client)  # type: ignore[arg-type]
        processed = await processor.process_results(batch_id, results, "anthropic")

        # Convert back to JSONL format
        processed_lines = []
        for p in processed:
            processed_lines.append(json.dumps(p.result))
            if p.was_processed:
                logger.info(
                    f"CCR: Processed batch result {p.custom_id} "
                    f"({p.continuation_rounds} continuation rounds)"
                )

        processed_content = "\n".join(processed_lines)

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="anthropic",
            model="batch:ccr-processed",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        return Response(
            content=processed_content.encode("utf-8"),
            status_code=200,
            media_type="application/jsonl",
        )
