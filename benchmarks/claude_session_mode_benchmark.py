#!/usr/bin/env python3
"""Replay real Claude Code sessions through baseline/token/cache simulations."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from headroom.cache.compression_cache import CompressionCache
from headroom.cache.prefix_tracker import PrefixCacheTracker
from headroom.pricing.litellm_pricing import get_model_pricing
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.models import ProxyConfig
from headroom.proxy.modes import PROXY_MODE_CACHE, PROXY_MODE_TOKEN
from headroom.proxy.server import HeadroomProxy
from headroom.tokenizers import get_tokenizer
from headroom.utils import extract_user_query

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
DEFAULT_CACHE_TTL_MINUTES = 5
OUTPUT_MD = "claude_session_mode_simulation.md"
OUTPUT_JSON = "claude_session_mode_simulation.json"
OUTPUT_HTML = "claude_session_mode_simulation.html"
CHECKPOINT_DIRNAME = "checkpoints"


@dataclass
class ReplayTurn:
    session_id: str
    project_key: str
    decoded_project_path: str
    request_id: str
    model: str
    timestamp: datetime
    input_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    output_tokens: int
    observed_input_tokens: int = 0
    observed_cache_read_tokens: int = 0
    observed_cache_write_tokens: int = 0


@dataclass
class SessionReplay:
    session_id: str
    project_key: str
    decoded_project_path: str
    turns: list[ReplayTurn] = field(default_factory=list)


@dataclass
class TurnMetrics:
    session_id: str
    request_id: str
    model: str
    timestamp: str
    raw_input_tokens: int
    forwarded_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    regular_input_tokens: int
    output_tokens: int
    paid_input_cost_usd: float
    cache_read_cost_usd: float
    cache_write_cost_usd: float
    paid_output_cost_usd: float
    total_cost_usd: float


@dataclass
class ModeSummary:
    mode: str
    sessions: int = 0
    requests: int = 0
    raw_input_tokens: int = 0
    forwarded_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    regular_input_tokens: int = 0
    output_tokens: int = 0
    paid_input_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0
    paid_output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    turns: list[TurnMetrics] = field(default_factory=list)

    @property
    def raw_tokens(self) -> int:
        return self.raw_input_tokens + self.output_tokens

    @property
    def cache_tokens(self) -> int:
        return self.cache_read_tokens + self.cache_write_tokens

    @property
    def prompt_window_with_cache(self) -> int:
        return self.forwarded_input_tokens

    @property
    def prompt_window_without_cache_reads(self) -> int:
        return self.forwarded_input_tokens - self.cache_read_tokens


@dataclass
class DatasetSummary:
    projects: int
    sessions: int
    requests: int
    models: dict[str, int]
    decoded_project_paths: int


@dataclass
class ObservedSummary:
    sessions: int = 0
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0
    paid_input_cost_usd: float = 0.0
    paid_output_cost_usd: float = 0.0
    healthy_growth_turns: int = 0
    broken_prefix_turns: int = 0
    resume_like_resets: int = 0

    @property
    def raw_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.output_tokens
        )

    @property
    def cache_ratio_pct(self) -> float:
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        if total <= 0:
            return 0.0
        return self.cache_read_tokens / total * 100.0


def _update_dataset_with_replay(
    dataset: DatasetSummary | None, replay: SessionReplay
) -> DatasetSummary:
    if dataset is None:
        dataset = DatasetSummary(
            projects=0,
            sessions=0,
            requests=0,
            models={},
            decoded_project_paths=0,
        )
    projects = {replay.project_key}
    project_paths = {replay.decoded_project_path}
    model_counts = Counter(dataset.models)
    requests = dataset.requests
    for turn in replay.turns:
        model_counts[turn.model] += 1
        requests += 1
    return DatasetSummary(
        projects=dataset.projects + len(projects),
        sessions=dataset.sessions + 1,
        requests=requests,
        models=dict(sorted(model_counts.items())),
        decoded_project_paths=dataset.decoded_project_paths + len(project_paths),
    )


def _turn_metrics_from_dict(data: dict[str, Any]) -> TurnMetrics:
    return TurnMetrics(**data)


def _mode_summary_from_dict(data: dict[str, Any]) -> ModeSummary:
    turns = [_turn_metrics_from_dict(turn) for turn in data.get("turns", [])]
    summary = ModeSummary(
        mode=data["mode"],
        sessions=data.get("sessions", 0),
        requests=data.get("requests", 0),
        raw_input_tokens=data.get("raw_input_tokens", 0),
        forwarded_input_tokens=data.get("forwarded_input_tokens", 0),
        cache_read_tokens=data.get("cache_read_tokens", 0),
        cache_write_tokens=data.get("cache_write_tokens", 0),
        regular_input_tokens=data.get("regular_input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        paid_input_cost_usd=data.get("paid_input_cost_usd", 0.0),
        cache_read_cost_usd=data.get("cache_read_cost_usd", 0.0),
        cache_write_cost_usd=data.get("cache_write_cost_usd", 0.0),
        paid_output_cost_usd=data.get("paid_output_cost_usd", 0.0),
        total_cost_usd=data.get("total_cost_usd", 0.0),
        turns=turns,
    )
    return summary


def decode_project_key(project_key: str) -> str:
    """Decode Claude's project directory encoding back to a local path-ish string."""
    if "--" not in project_key:
        return project_key.replace("-", "\\")
    drive, remainder = project_key.split("--", 1)
    return drive + ":\\" + remainder.replace("-", "\\")


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(UTC)


def _canonical_block_key(block: Any) -> str:
    return json.dumps(block, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _assistant_blocks_from_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                return True
    return False


def _finalize_group(
    group: dict[str, Any] | None,
    pending_messages: list[dict[str, Any]],
    turns: list[ReplayTurn],
    *,
    session_id: str,
    project_key: str,
    decoded_project_path: str,
) -> None:
    if not group:
        return
    assistant_message = {
        "role": "assistant",
        "content": group["blocks"] if group["blocks"] else "",
    }
    turns.append(
        ReplayTurn(
            session_id=session_id,
            project_key=project_key,
            decoded_project_path=decoded_project_path,
            request_id=group["request_id"],
            model=group["model"],
            timestamp=group["timestamp"],
            input_messages=copy.deepcopy(pending_messages),
            assistant_message=assistant_message,
            output_tokens=group["output_tokens"],
            observed_input_tokens=group["observed_input_tokens"],
            observed_cache_read_tokens=group["observed_cache_read_tokens"],
            observed_cache_write_tokens=group["observed_cache_write_tokens"],
        )
    )


def load_session_replay(session_file: Path) -> SessionReplay | None:
    """Load a top-level Claude session transcript into replayable request turns."""
    project_key = session_file.parent.name
    decoded_project_path = decode_project_key(project_key)
    session_id = session_file.stem
    pending_messages: list[dict[str, Any]] = []
    turns: list[ReplayTurn] = []
    current_group: dict[str, Any] | None = None

    with session_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            message = event.get("message")

            if event_type == "user" and isinstance(message, dict) and message.get("role") == "user":
                _finalize_group(
                    current_group,
                    pending_messages,
                    turns,
                    session_id=session_id,
                    project_key=project_key,
                    decoded_project_path=decoded_project_path,
                )
                current_group = None
                pending_messages.clear()
                pending_messages.append(copy.deepcopy(message))
                continue

            if (
                event_type == "assistant"
                and isinstance(message, dict)
                and message.get("role") == "assistant"
                and event.get("requestId")
            ):
                request_id = str(event["requestId"])
                usage = message.get("usage") or {}
                timestamp = _parse_timestamp(event.get("timestamp"))
                blocks = _assistant_blocks_from_content(message.get("content"))
                if current_group is None or current_group["request_id"] != request_id:
                    had_group = current_group is not None
                    _finalize_group(
                        current_group,
                        pending_messages,
                        turns,
                        session_id=session_id,
                        project_key=project_key,
                        decoded_project_path=decoded_project_path,
                    )
                    if had_group:
                        pending_messages.clear()
                    current_group = {
                        "request_id": request_id,
                        "model": str(message.get("model", "unknown")),
                        "timestamp": timestamp,
                        "blocks": [],
                        "seen": set(),
                        "output_tokens": 0,
                        "observed_input_tokens": 0,
                        "observed_cache_read_tokens": 0,
                        "observed_cache_write_tokens": 0,
                    }
                for block in blocks:
                    key = _canonical_block_key(block)
                    if key not in current_group["seen"]:
                        current_group["seen"].add(key)
                        current_group["blocks"].append(copy.deepcopy(block))
                current_group["output_tokens"] = max(
                    current_group["output_tokens"],
                    int(usage.get("output_tokens", 0) or 0),
                )
                current_group["observed_input_tokens"] = max(
                    current_group["observed_input_tokens"],
                    int(usage.get("input_tokens", 0) or 0),
                )
                current_group["observed_cache_read_tokens"] = max(
                    current_group["observed_cache_read_tokens"],
                    int(usage.get("cache_read_input_tokens", 0) or 0),
                )
                current_group["observed_cache_write_tokens"] = max(
                    current_group["observed_cache_write_tokens"],
                    int(usage.get("cache_creation_input_tokens", 0) or 0),
                )

    _finalize_group(
        current_group,
        pending_messages,
        turns,
        session_id=session_id,
        project_key=project_key,
        decoded_project_path=decoded_project_path,
    )

    if not turns:
        return None
    return SessionReplay(
        session_id=session_id,
        project_key=project_key,
        decoded_project_path=decoded_project_path,
        turns=turns,
    )


def discover_session_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for project_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files.extend(
            sorted(p for p in project_dir.iterdir() if p.is_file() and p.suffix == ".jsonl")
        )
    return files


def load_replays(root: Path, max_sessions: int | None = None) -> list[SessionReplay]:
    replays: list[SessionReplay] = []
    session_files = discover_session_files(root)
    total = len(session_files)
    for index, session_file in enumerate(session_files, start=1):
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[load] session={index}/{total} file={session_file.name}", flush=True)
        replay = load_session_replay(session_file)
        if replay is not None:
            replays.append(replay)
        if max_sessions is not None and len(replays) >= max_sessions:
            break
    return replays


def select_session_files(root: Path, max_sessions: int | None = None) -> list[Path]:
    session_files = discover_session_files(root)
    if max_sessions is not None:
        session_files = session_files[:max_sessions]
    return session_files


def build_dataset_and_observed_from_files(
    session_files: list[Path], *, cache_write_multiplier: float = 1.25
) -> tuple[DatasetSummary, ObservedSummary]:
    model_counts: Counter[str] = Counter()
    project_keys: set[str] = set()
    decoded_project_paths: set[str] = set()
    requests = 0
    observed = ObservedSummary()

    total = len(session_files)
    for index, session_file in enumerate(session_files, start=1):
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[load] session={index}/{total} file={session_file.name}", flush=True)
        replay = load_session_replay(session_file)
        if replay is None:
            continue
        project_keys.add(replay.project_key)
        decoded_project_paths.add(replay.decoded_project_path)
        observed.sessions += 1
        for turn in replay.turns:
            model_counts[turn.model] += 1
            requests += 1
            rates = _resolve_model_rates(turn.model, cache_write_multiplier=cache_write_multiplier)
            observed.requests += 1
            observed.input_tokens += turn.observed_input_tokens
            observed.cache_read_tokens += turn.observed_cache_read_tokens
            observed.cache_write_tokens += turn.observed_cache_write_tokens
            observed.output_tokens += turn.output_tokens
            observed.paid_input_cost_usd += turn.observed_input_tokens * rates["input"]
            observed.cache_read_cost_usd += turn.observed_cache_read_tokens * rates["cache_read"]
            observed.cache_write_cost_usd += turn.observed_cache_write_tokens * rates["cache_write"]
            observed.paid_output_cost_usd += turn.output_tokens * rates["output"]

        prev_read = 0
        prev_write = 0
        for turn in replay.turns:
            read = turn.observed_cache_read_tokens
            write = turn.observed_cache_write_tokens
            if read > prev_read and write <= prev_write:
                observed.healthy_growth_turns += 1
            if read == prev_read and write > prev_write:
                observed.broken_prefix_turns += 1
            if read < prev_read and write > 0:
                observed.resume_like_resets += 1
            prev_read = read
            prev_write = write

    observed.total_cost_usd = (
        observed.paid_input_cost_usd
        + observed.cache_read_cost_usd
        + observed.cache_write_cost_usd
        + observed.paid_output_cost_usd
    )
    dataset = DatasetSummary(
        projects=len(project_keys),
        sessions=observed.sessions,
        requests=requests,
        models=dict(sorted(model_counts.items())),
        decoded_project_paths=len(decoded_project_paths),
    )
    return dataset, observed


def summarize_dataset(replays: list[SessionReplay]) -> DatasetSummary:
    model_counts: Counter[str] = Counter()
    project_paths: set[str] = set()
    requests = 0
    for replay in replays:
        project_paths.add(replay.decoded_project_path)
        for turn in replay.turns:
            model_counts[turn.model] += 1
            requests += 1
    return DatasetSummary(
        projects=len({r.project_key for r in replays}),
        sessions=len(replays),
        requests=requests,
        models=dict(sorted(model_counts.items())),
        decoded_project_paths=len(project_paths),
    )


def summarize_observed_usage(
    replays: list[SessionReplay], *, cache_write_multiplier: float = 1.25
) -> ObservedSummary:
    summary = ObservedSummary(sessions=len(replays))
    for replay in replays:
        prev_read = 0
        prev_write = 0
        for turn in replay.turns:
            rates = _resolve_model_rates(turn.model, cache_write_multiplier=cache_write_multiplier)
            summary.requests += 1
            summary.input_tokens += turn.observed_input_tokens
            summary.cache_read_tokens += turn.observed_cache_read_tokens
            summary.cache_write_tokens += turn.observed_cache_write_tokens
            summary.output_tokens += turn.output_tokens

            summary.paid_input_cost_usd += turn.observed_input_tokens * rates["input"]
            summary.cache_read_cost_usd += turn.observed_cache_read_tokens * rates["cache_read"]
            summary.cache_write_cost_usd += turn.observed_cache_write_tokens * rates["cache_write"]
            summary.paid_output_cost_usd += turn.output_tokens * rates["output"]

            read = turn.observed_cache_read_tokens
            write = turn.observed_cache_write_tokens
            if read > prev_read and write <= prev_write:
                summary.healthy_growth_turns += 1
            if read == prev_read and write > prev_write:
                summary.broken_prefix_turns += 1
            if read < prev_read and write > 0:
                summary.resume_like_resets += 1
            prev_read = read
            prev_write = write

    summary.total_cost_usd = (
        summary.paid_input_cost_usd
        + summary.cache_read_cost_usd
        + summary.cache_write_cost_usd
        + summary.paid_output_cost_usd
    )
    return summary


def _common_prefix_tokens(
    prev: list[dict[str, Any]],
    curr: list[dict[str, Any]],
    tokenizer: Any,
) -> int:
    common = 0
    for a, b in zip(prev, curr):
        if a != b:
            break
        common += tokenizer.count_message(b)
    return common


def _make_proxy(mode: str) -> HeadroomProxy:
    cfg = ProxyConfig(
        mode=mode,
        optimize=True,
        image_optimize=True,
        smart_routing=False,
        code_aware_enabled=False,
        read_lifecycle=False,
        intelligent_context=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
    )
    return HeadroomProxy(cfg)


def _apply_mode_to_messages(
    proxy: HeadroomProxy | None,
    mode: str,
    messages: list[dict[str, Any]],
    *,
    model: str,
    prefix_tracker: PrefixCacheTracker | None,
    comp_cache: CompressionCache | None,
) -> list[dict[str, Any]]:
    if mode == "baseline":
        return copy.deepcopy(messages)

    assert proxy is not None
    assert prefix_tracker is not None
    frozen_message_count = prefix_tracker.get_frozen_message_count()
    if mode == PROXY_MODE_CACHE:
        frozen_message_count = AnthropicHandlerMixin._strict_previous_turn_frozen_count(
            messages,
            frozen_message_count,
        )

    working_messages = copy.deepcopy(messages)
    if proxy.config.image_optimize and working_messages and _messages_have_images(working_messages):
        from headroom.proxy.helpers import _get_image_compressor

        compressor = _get_image_compressor()
        if compressor and compressor.has_images(working_messages):
            if mode == PROXY_MODE_CACHE:
                working_messages = (
                    AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
                        working_messages,
                        frozen_message_count=frozen_message_count,
                        compressor=compressor,
                    )
                )
            else:
                working_messages = compressor.compress(working_messages, provider="anthropic")

    if mode == PROXY_MODE_TOKEN and comp_cache is not None:
        working_messages = comp_cache.apply_cached(working_messages)
        cache_frozen_count = comp_cache.compute_frozen_count(messages)
        frozen_message_count = min(frozen_message_count, cache_frozen_count)

    context_limit = proxy.anthropic_provider.get_context_limit(model)
    result = proxy.anthropic_pipeline.apply(
        messages=working_messages,
        model=model,
        model_limit=context_limit,
        context=extract_user_query(working_messages),
        frozen_message_count=frozen_message_count,
    )
    forwarded = result.messages

    if mode == PROXY_MODE_TOKEN and comp_cache is not None and forwarded != working_messages:
        comp_cache.update_from_result(messages, forwarded)
    if mode == PROXY_MODE_CACHE:
        forwarded, _ = AnthropicHandlerMixin._restore_frozen_prefix(
            messages,
            forwarded,
            frozen_message_count=frozen_message_count,
        )
    return forwarded


@dataclass
class _PendingTurn:
    summary: ModeSummary
    turn: ReplayTurn
    tokenizer: Any
    raw_input_tokens: int
    forwarded: list[dict[str, Any]]


def _cache_gap_within_ttl(
    current_ts: datetime,
    previous_ts: datetime | None,
    *,
    ttl: timedelta,
) -> bool:
    if previous_ts is None:
        return False
    return current_ts - previous_ts <= ttl


def _resolve_model_rates(model: str, *, cache_write_multiplier: float) -> dict[str, float]:
    pricing = get_model_pricing(model)
    if pricing is None:
        if "opus" in model:
            input_per_1m = 15.0
            output_per_1m = 75.0
        elif "haiku" in model:
            input_per_1m = 1.0
            output_per_1m = 5.0
        else:
            input_per_1m = 3.0
            output_per_1m = 15.0
    else:
        input_per_1m = pricing.input_cost_per_1m
        output_per_1m = pricing.output_cost_per_1m
    return {
        "input": input_per_1m / 1_000_000,
        "output": output_per_1m / 1_000_000,
        "cache_read": (input_per_1m * 0.10) / 1_000_000,
        "cache_write": (input_per_1m * cache_write_multiplier) / 1_000_000,
    }


def _apply_turn_metrics(
    summary: ModeSummary,
    turn: ReplayTurn,
    *,
    raw_input_tokens: int,
    tokenizer: Any,
    forwarded: list[dict[str, Any]],
    previous_forwarded: list[dict[str, Any]],
    previous_timestamp: datetime | None,
    next_forwarded: list[dict[str, Any]] | None,
    next_timestamp: datetime | None,
    ttl: timedelta,
    cache_write_multiplier: float,
) -> None:
    forwarded_input_tokens = tokenizer.count_messages(forwarded)

    read_tokens = 0
    if _cache_gap_within_ttl(turn.timestamp, previous_timestamp, ttl=ttl):
        read_tokens = _common_prefix_tokens(previous_forwarded, forwarded, tokenizer)

    write_tokens = 0
    if next_forwarded is not None and _cache_gap_within_ttl(
        next_timestamp, turn.timestamp, ttl=ttl
    ):
        next_common = _common_prefix_tokens(forwarded, next_forwarded, tokenizer)
        write_tokens = max(0, next_common - read_tokens)

    regular_input_tokens = max(0, forwarded_input_tokens - read_tokens - write_tokens)
    rates = _resolve_model_rates(turn.model, cache_write_multiplier=cache_write_multiplier)
    paid_input_cost_usd = regular_input_tokens * rates["input"]
    cache_read_cost_usd = read_tokens * rates["cache_read"]
    cache_write_cost_usd = write_tokens * rates["cache_write"]
    paid_output_cost_usd = turn.output_tokens * rates["output"]
    total_cost_usd = (
        paid_input_cost_usd + cache_read_cost_usd + cache_write_cost_usd + paid_output_cost_usd
    )

    summary.requests += 1
    summary.raw_input_tokens += raw_input_tokens
    summary.forwarded_input_tokens += forwarded_input_tokens
    summary.cache_read_tokens += read_tokens
    summary.cache_write_tokens += write_tokens
    summary.regular_input_tokens += regular_input_tokens
    summary.output_tokens += turn.output_tokens
    summary.paid_input_cost_usd += paid_input_cost_usd
    summary.cache_read_cost_usd += cache_read_cost_usd
    summary.cache_write_cost_usd += cache_write_cost_usd
    summary.paid_output_cost_usd += paid_output_cost_usd
    summary.total_cost_usd += total_cost_usd


def _merge_mode_summary(target: ModeSummary, source: ModeSummary) -> None:
    target.sessions += source.sessions
    target.requests += source.requests
    target.raw_input_tokens += source.raw_input_tokens
    target.forwarded_input_tokens += source.forwarded_input_tokens
    target.cache_read_tokens += source.cache_read_tokens
    target.cache_write_tokens += source.cache_write_tokens
    target.regular_input_tokens += source.regular_input_tokens
    target.output_tokens += source.output_tokens
    target.paid_input_cost_usd += source.paid_input_cost_usd
    target.cache_read_cost_usd += source.cache_read_cost_usd
    target.cache_write_cost_usd += source.cache_write_cost_usd
    target.paid_output_cost_usd += source.paid_output_cost_usd
    target.total_cost_usd += source.total_cost_usd


def _disable_headroom_benchmark_logging() -> None:
    logging.raiseExceptions = False
    for logger_name in (
        "headroom",
        "headroom.cache",
        "headroom.cache.compression_cache",
        "headroom.proxy",
        "headroom.transforms",
    ):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)


def _checkpoint_path(checkpoint_dir: Path, mode: str, replay: SessionReplay) -> Path:
    return checkpoint_dir / f"{mode}--{replay.session_id}.json"


def _checkpoint_path_for_session_id(checkpoint_dir: Path, mode: str, session_id: str) -> Path:
    return checkpoint_dir / f"{mode}--{session_id}.json"


def _load_checkpoint(checkpoint_dir: Path, mode: str, replay: SessionReplay) -> ModeSummary | None:
    path = _checkpoint_path(checkpoint_dir, mode, replay)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _mode_summary_from_dict(payload)


def _load_checkpoint_by_session_id(
    checkpoint_dir: Path, mode: str, session_id: str
) -> ModeSummary | None:
    path = _checkpoint_path_for_session_id(checkpoint_dir, mode, session_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _mode_summary_from_dict(payload)


def _write_checkpoint(
    checkpoint_dir: Path,
    mode: str,
    replay: SessionReplay,
    summary: ModeSummary,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, mode, replay)
    payload = asdict(summary)
    payload["turns"] = []
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_checkpoint_by_session_id(
    checkpoint_dir: Path, mode: str, session_id: str, summary: ModeSummary
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path_for_session_id(checkpoint_dir, mode, session_id)
    payload = asdict(summary)
    payload["turns"] = []
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _simulate_single_replay_mode(
    replay: SessionReplay,
    mode: str,
    cache_ttl_minutes: int,
    cache_write_multiplier: float,
) -> ModeSummary:
    _disable_headroom_benchmark_logging()

    summary = ModeSummary(mode=mode, sessions=1)
    ttl = timedelta(minutes=cache_ttl_minutes)
    proxy = None if mode == "baseline" else _make_proxy(mode)
    pending: _PendingTurn | None = None
    conversation: list[dict[str, Any]] = []
    conversation_token_total = 0
    previous_forwarded: list[dict[str, Any]] = []
    previous_timestamp: datetime | None = None
    prefix_tracker = None if mode == "baseline" else PrefixCacheTracker("anthropic")
    comp_cache = CompressionCache() if mode == PROXY_MODE_TOKEN else None

    for turn in replay.turns:
        tokenizer = get_tokenizer(turn.model)
        turn_input_token_total = sum(tokenizer.count_message(msg) for msg in turn.input_messages)
        conversation.extend(turn.input_messages)
        raw_input_tokens = conversation_token_total + turn_input_token_total
        forwarded = _apply_mode_to_messages(
            proxy,
            mode,
            conversation,
            model=turn.model,
            prefix_tracker=prefix_tracker,
            comp_cache=comp_cache,
        )
        if pending is not None:
            _apply_turn_metrics(
                pending.summary,
                pending.turn,
                raw_input_tokens=pending.raw_input_tokens,
                tokenizer=pending.tokenizer,
                forwarded=pending.forwarded,
                previous_forwarded=previous_forwarded,
                previous_timestamp=previous_timestamp,
                next_forwarded=forwarded,
                next_timestamp=turn.timestamp,
                ttl=ttl,
                cache_write_multiplier=cache_write_multiplier,
            )
            previous_forwarded = copy.deepcopy(pending.forwarded)
            previous_timestamp = pending.turn.timestamp

        if prefix_tracker is not None:
            prefix_tracker.update_from_response(
                cache_read_tokens=0,
                cache_write_tokens=0,
                messages=forwarded,
                message_token_counts=[tokenizer.count_message(msg) for msg in forwarded],
            )

        pending = _PendingTurn(
            summary=summary,
            turn=turn,
            tokenizer=tokenizer,
            raw_input_tokens=raw_input_tokens,
            forwarded=forwarded,
        )
        conversation.append(turn.assistant_message)
        conversation_token_total = (
            raw_input_tokens + tokenizer.count_message(turn.assistant_message)
        )

    if pending is not None:
        _apply_turn_metrics(
            pending.summary,
            pending.turn,
            raw_input_tokens=pending.raw_input_tokens,
            tokenizer=pending.tokenizer,
            forwarded=pending.forwarded,
            previous_forwarded=previous_forwarded,
            previous_timestamp=previous_timestamp,
            next_forwarded=None,
            next_timestamp=None,
            ttl=ttl,
            cache_write_multiplier=cache_write_multiplier,
        )

    return summary


def _simulate_single_session_file_mode(
    session_file: Path,
    mode: str,
    cache_ttl_minutes: int,
    cache_write_multiplier: float,
) -> tuple[str, ModeSummary]:
    replay = load_session_replay(session_file)
    if replay is None:
        return session_file.stem, ModeSummary(mode=mode)
    return replay.session_id, _simulate_single_replay_mode(
        replay,
        mode,
        cache_ttl_minutes,
        cache_write_multiplier,
    )


def simulate_replays(
    replays: list[SessionReplay],
    *,
    cache_ttl_minutes: int = DEFAULT_CACHE_TTL_MINUTES,
    cache_write_multiplier: float = 1.25,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> tuple[DatasetSummary, dict[str, ModeSummary]]:
    dataset = summarize_dataset(replays)
    summaries = {
        "baseline": ModeSummary(mode="baseline"),
        PROXY_MODE_TOKEN: ModeSummary(mode=PROXY_MODE_TOKEN),
        PROXY_MODE_CACHE: ModeSummary(mode=PROXY_MODE_CACHE),
    }

    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        print(f"[simulate] mode={mode} sessions={len(replays)}", flush=True)
        worker_count = workers if workers > 0 else max(1, min(8, os.cpu_count() or 1))
        if worker_count > 1 and len(replays) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map: dict[concurrent.futures.Future[ModeSummary], SessionReplay] = {}
                completed = 0
                for replay in replays:
                    cached = (
                        _load_checkpoint(checkpoint_dir, mode, replay)
                        if checkpoint_dir is not None
                        else None
                    )
                    if cached is not None:
                        completed += 1
                        _merge_mode_summary(summaries[mode], cached)
                        if completed == 1 or completed % 10 == 0 or completed == len(replays):
                            print(
                                f"[simulate] mode={mode} completed={completed}/{len(replays)}",
                                flush=True,
                            )
                        continue
                    future = executor.submit(
                        _simulate_single_replay_mode,
                        replay,
                        mode,
                        cache_ttl_minutes,
                        cache_write_multiplier,
                    )
                    future_map[future] = replay
                for future in concurrent.futures.as_completed(future_map):
                    replay = future_map[future]
                    partial = future.result()
                    if checkpoint_dir is not None:
                        _write_checkpoint(checkpoint_dir, mode, replay, partial)
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == len(replays):
                        print(
                            f"[simulate] mode={mode} completed={completed}/{len(replays)}",
                            flush=True,
                        )
                    _merge_mode_summary(summaries[mode], partial)
        else:
            for index, replay in enumerate(replays, start=1):
                cached = (
                    _load_checkpoint(checkpoint_dir, mode, replay)
                    if checkpoint_dir is not None
                    else None
                )
                if cached is not None:
                    _merge_mode_summary(summaries[mode], cached)
                    continue
                if index == 1 or index % 10 == 0 or index == len(replays):
                    print(
                        f"[simulate] mode={mode} session={index}/{len(replays)} "
                        f"requests={len(replay.turns)}",
                        flush=True,
                    )
                partial = _simulate_single_replay_mode(
                    replay,
                    mode,
                    cache_ttl_minutes,
                    cache_write_multiplier,
                )
                if checkpoint_dir is not None:
                    _write_checkpoint(checkpoint_dir, mode, replay, partial)
                _merge_mode_summary(summaries[mode], partial)

    return dataset, summaries


def simulate_session_files(
    session_files: list[Path],
    dataset: DatasetSummary,
    *,
    cache_ttl_minutes: int = DEFAULT_CACHE_TTL_MINUTES,
    cache_write_multiplier: float = 1.25,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> dict[str, ModeSummary]:
    summaries = {
        "baseline": ModeSummary(mode="baseline"),
        PROXY_MODE_TOKEN: ModeSummary(mode=PROXY_MODE_TOKEN),
        PROXY_MODE_CACHE: ModeSummary(mode=PROXY_MODE_CACHE),
    }
    total = len(session_files)

    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        print(f"[simulate] mode={mode} sessions={total}", flush=True)
        worker_count = workers if workers > 0 else 1
        if worker_count > 1 and total > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_disable_headroom_benchmark_logging,
            ) as executor:
                future_map: dict[concurrent.futures.Future[tuple[str, ModeSummary]], str] = {}
                completed = 0
                for session_file in session_files:
                    session_id = session_file.stem
                    cached = (
                        _load_checkpoint_by_session_id(checkpoint_dir, mode, session_id)
                        if checkpoint_dir is not None
                        else None
                    )
                    if cached is not None:
                        completed += 1
                        _merge_mode_summary(summaries[mode], cached)
                        if completed == 1 or completed % 10 == 0 or completed == total:
                            print(
                                f"[simulate] mode={mode} completed={completed}/{total}",
                                flush=True,
                            )
                        continue
                    future = executor.submit(
                        _simulate_single_session_file_mode,
                        session_file,
                        mode,
                        cache_ttl_minutes,
                        cache_write_multiplier,
                    )
                    future_map[future] = session_id
                for future in concurrent.futures.as_completed(future_map):
                    session_id, partial = future.result()
                    if checkpoint_dir is not None:
                        _write_checkpoint_by_session_id(checkpoint_dir, mode, session_id, partial)
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == total:
                        print(
                            f"[simulate] mode={mode} completed={completed}/{total}",
                            flush=True,
                        )
                    _merge_mode_summary(summaries[mode], partial)
        else:
            for index, session_file in enumerate(session_files, start=1):
                session_id = session_file.stem
                cached = (
                    _load_checkpoint_by_session_id(checkpoint_dir, mode, session_id)
                    if checkpoint_dir is not None
                    else None
                )
                if cached is not None:
                    _merge_mode_summary(summaries[mode], cached)
                    if index == 1 or index % 10 == 0 or index == total:
                        print(
                            f"[simulate] mode={mode} completed={index}/{total}",
                            flush=True,
                        )
                    continue
                replay = load_session_replay(session_file)
                if replay is None:
                    continue
                if index == 1 or index % 10 == 0 or index == total:
                    print(
                        f"[simulate] mode={mode} session={index}/{total} "
                        f"requests={len(replay.turns)}",
                        flush=True,
                    )
                partial = _simulate_single_replay_mode(
                    replay,
                    mode,
                    cache_ttl_minutes,
                    cache_write_multiplier,
                )
                if checkpoint_dir is not None:
                    _write_checkpoint_by_session_id(checkpoint_dir, mode, session_id, partial)
                _merge_mode_summary(summaries[mode], partial)

    return summaries


def determine_winners(summaries: dict[str, ModeSummary]) -> dict[str, str]:
    return {
        "total_cost": min(summaries.values(), key=lambda s: s.total_cost_usd).mode,
        "window_with_cache": min(summaries.values(), key=lambda s: s.prompt_window_with_cache).mode,
        "window_without_cache_reads": min(
            summaries.values(), key=lambda s: s.prompt_window_without_cache_reads
        ).mode,
    }


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def print_console_report(dataset: DatasetSummary, summaries: dict[str, ModeSummary]) -> None:
    winners = determine_winners(summaries)
    print("Claude session mode simulation")
    print(
        f"Dataset: {dataset.projects} projects, {dataset.sessions} sessions, "
        f"{dataset.requests} requests"
    )
    print()
    print(
        "mode      raw_tok      cache_tok    cache_read   cache_write   paid_in      paid_out     total_cost"
    )
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        print(
            f"{mode:<9} {summary.raw_tokens:>11,} {summary.cache_tokens:>12,} "
            f"{summary.cache_read_tokens:>11,} {summary.cache_write_tokens:>12,} "
            f"{summary.regular_input_tokens:>10,} {summary.output_tokens:>12,} "
            f"{format_currency(summary.total_cost_usd):>11}"
        )
    print()
    print(f"Winner by total cost: {winners['total_cost']}")
    print(f"Winner if cache tokens count against window: {winners['window_with_cache']}")
    print(
        "Winner if cache read tokens do not count against window: "
        f"{winners['window_without_cache_reads']}"
    )


def print_observed_console_report(observed: ObservedSummary) -> None:
    print()
    print("Observed Claude session usage")
    print(
        f"requests={observed.requests:,} cache_ratio={observed.cache_ratio_pct:.1f}% "
        f"broken_prefix_turns={observed.broken_prefix_turns:,} "
        f"resume_like_resets={observed.resume_like_resets:,}"
    )
    print(
        f"input={observed.input_tokens:,} cache_read={observed.cache_read_tokens:,} "
        f"cache_write={observed.cache_write_tokens:,} output={observed.output_tokens:,} "
        f"total_cost={format_currency(observed.total_cost_usd)}"
    )


def build_report_markdown(
    dataset: DatasetSummary,
    observed: ObservedSummary,
    summaries: dict[str, ModeSummary],
) -> str:
    winners = determine_winners(summaries)
    model_lines = "\n".join(f"- `{model}`: {count}" for model, count in dataset.models.items())
    rows = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        rows.append(
            "| "
            + " | ".join(
                [
                    summary.mode,
                    f"{summary.raw_tokens:,}",
                    f"{summary.cache_tokens:,}",
                    f"{summary.cache_read_tokens:,}",
                    f"{summary.cache_write_tokens:,}",
                    f"{summary.regular_input_tokens:,}",
                    f"{summary.output_tokens:,}",
                    format_currency(summary.paid_input_cost_usd),
                    format_currency(summary.cache_read_cost_usd),
                    format_currency(summary.cache_write_cost_usd),
                    format_currency(summary.paid_output_cost_usd),
                    format_currency(summary.total_cost_usd),
                    f"{summary.prompt_window_with_cache:,}",
                    f"{summary.prompt_window_without_cache_reads:,}",
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# Claude Session Mode Simulation",
            "",
            "## Dataset",
            "",
            f"- Projects: {dataset.projects}",
            f"- Sessions: {dataset.sessions}",
            f"- Requests: {dataset.requests}",
            f"- Distinct decoded project paths: {dataset.decoded_project_paths}",
            "- Models:",
            model_lines or "- None",
            "",
            "## Assumptions",
            "",
            "- Uses top-level session `.jsonl` files under `~/.claude/projects`.",
            "- Replays only transcript-visible messages. Hidden system/tool schemas from Claude Code are not available in local transcript files and are therefore excluded.",
            "- Simulates Anthropic prompt caching with a 5 minute TTL.",
            "- Estimates cache read cost as 10% of base input price and cache write/store cost as 125% of base input price.",
            "- Holds recorded output token counts constant across baseline/token/cache so comparisons isolate input-side behavior.",
            "",
            "## Observed",
            "",
            f"- Requests with observed usage: {observed.requests:,}",
            f"- Cache ratio: {observed.cache_ratio_pct:.1f}%",
            f"- Healthy growth turns: {observed.healthy_growth_turns:,}",
            f"- Broken prefix turns: {observed.broken_prefix_turns:,}",
            f"- Resume-like resets: {observed.resume_like_resets:,}",
            f"- Observed total cost: {format_currency(observed.total_cost_usd)}",
            "",
            "## Summary",
            "",
            "| Mode | Raw Tokens | Cache Tokens | Cache Read | Cache Write | Paid Input Tokens | Paid Output Tokens | Paid Input Cost | Cache Read Cost | Cache Write Cost | Paid Output Cost | Total Cost | Window Tokens (Cache Counted) | Window Tokens (Cache Reads Excluded) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Winners",
            "",
            f"- Total cost winner: `{winners['total_cost']}`",
            f"- Window winner if cache tokens count: `{winners['window_with_cache']}`",
            "- Window winner if cache read tokens do not count: "
            f"`{winners['window_without_cache_reads']}`",
        ]
    )


def build_report_html(
    dataset: DatasetSummary,
    observed: ObservedSummary,
    summaries: dict[str, ModeSummary],
) -> str:
    winners = determine_winners(summaries)
    model_items = "".join(
        f"<li><code>{model}</code><span>{count:,}</span></li>"
        for model, count in dataset.models.items()
    )
    summary_rows = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        summary_rows.append(
            "<tr>"
            f"<td><span class='badge'>{summary.mode}</span></td>"
            f"<td>{summary.raw_tokens:,}</td>"
            f"<td>{summary.cache_tokens:,}</td>"
            f"<td>{summary.cache_read_tokens:,}</td>"
            f"<td>{summary.cache_write_tokens:,}</td>"
            f"<td>{summary.regular_input_tokens:,}</td>"
            f"<td>{summary.output_tokens:,}</td>"
            f"<td>{format_currency(summary.total_cost_usd)}</td>"
            f"<td>{summary.prompt_window_with_cache:,}</td>"
            f"<td>{summary.prompt_window_without_cache_reads:,}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claude Session Mode Simulation</title>
  <style>
    :root {{
      --bg: #fafaf9;
      --fg: #0f172a;
      --muted: #64748b;
      --card: rgba(255,255,255,0.88);
      --border: #e2e8f0;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --warn: #b45309;
      --bad: #b91c1c;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      --radius: 18px;
      --font: "Geist", "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      color: var(--fg);
      background:
        radial-gradient(circle at top left, #dbeafe 0%, transparent 35%),
        radial-gradient(circle at top right, #ccfbf1 0%, transparent 30%),
        linear-gradient(180deg, #f8fafc 0%, #f8fafc 100%);
    }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 40px 20px 64px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,250,252,0.86));
      border: 1px solid rgba(226,232,240,0.9);
      box-shadow: var(--shadow);
      border-radius: 28px;
      padding: 28px;
      backdrop-filter: blur(12px);
    }}
    h1, h2 {{ margin: 0 0 12px; letter-spacing: -0.03em; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.55; }}
    .grid {{ display: grid; gap: 16px; margin-top: 20px; }}
    .grid.cards {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(226,232,240,0.95);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .eyebrow {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 10px; }}
    .subtle {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
    .section {{ margin-top: 22px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(248,250,252,0.8); }}
    .badge {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 10px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent); font-weight: 600; font-size: 12px;
    }}
    ul.models {{ list-style: none; padding: 0; margin: 0; }}
    ul.models li {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); }}
    .winner-list div {{ margin-top: 10px; font-size: 15px; }}
    .good {{ color: var(--accent); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .95em; }}
    @media (max-width: 720px) {{
      .shell {{ padding: 20px 12px 40px; }}
      .hero {{ padding: 20px; border-radius: 22px; }}
      .value {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Local Claude Cache Analysis</div>
      <h1>Claude Session Mode Simulation</h1>
      <p>Observed usage is read directly from <code>~/.claude/projects</code>. Baseline, token, and cache are replayed locally through Headroom without making API calls.</p>
      <div class="grid cards">
        <div class="card"><div class="eyebrow">Projects</div><div class="value">{dataset.projects:,}</div><div class="subtle">{dataset.sessions:,} sessions / {dataset.requests:,} requests</div></div>
        <div class="card"><div class="eyebrow">Observed Cache Ratio</div><div class="value">{observed.cache_ratio_pct:.1f}%</div><div class="subtle">read / (read + write + input)</div></div>
        <div class="card"><div class="eyebrow">Observed Total Cost</div><div class="value">{format_currency(observed.total_cost_usd)}</div><div class="subtle">{observed.cache_read_tokens:,} read / {observed.cache_write_tokens:,} write</div></div>
        <div class="card"><div class="eyebrow">Broken Prefix Turns</div><div class="value">{observed.broken_prefix_turns:,}</div><div class="subtle">CR stuck while CC grows</div></div>
      </div>
    </section>
    <section class="section grid" style="grid-template-columns: 1.1fr .9fr;">
      <div class="card">
        <h2>Winners</h2>
        <div class="winner-list">
          <div><span class="eyebrow">Total cost</span><br><span class="badge">{winners["total_cost"]}</span></div>
          <div><span class="eyebrow">Window if cache counts</span><br><span class="badge">{winners["window_with_cache"]}</span></div>
          <div><span class="eyebrow">Window if cache reads do not count</span><br><span class="badge">{winners["window_without_cache_reads"]}</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Models</h2>
        <ul class="models">{model_items}</ul>
      </div>
    </section>
    <section class="section card">
      <h2>Observed Diagnostics</h2>
      <div class="grid cards">
        <div><div class="eyebrow">Healthy Growth Turns</div><div class="value good">{observed.healthy_growth_turns:,}</div></div>
        <div><div class="eyebrow">Broken Prefix Turns</div><div class="value bad">{observed.broken_prefix_turns:,}</div></div>
        <div><div class="eyebrow">Resume-like Resets</div><div class="value warn">{observed.resume_like_resets:,}</div></div>
      </div>
    </section>
    <section class="section card">
      <h2>Mode Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Mode</th><th>Raw Tokens</th><th>Cache Tokens</th><th>Cache Read</th><th>Cache Write</th><th>Paid Input</th><th>Paid Output</th><th>Total Cost</th><th>Window With Cache</th><th>Window Without Cache Reads</th>
            </tr>
          </thead>
          <tbody>
            {"".join(summary_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>"""


def write_report(
    output_dir: Path,
    dataset: DatasetSummary,
    observed: ObservedSummary,
    summaries: dict[str, ModeSummary],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / OUTPUT_MD
    json_path = output_dir / OUTPUT_JSON
    html_path = output_dir / OUTPUT_HTML
    md_path.write_text(build_report_markdown(dataset, observed, summaries), encoding="utf-8")
    html_path.write_text(build_report_html(dataset, observed, summaries), encoding="utf-8")
    payload = {
        "dataset": asdict(dataset),
        "observed": asdict(observed),
        "summaries": {mode: asdict(summary) for mode, summary in summaries.items()},
        "winners": determine_winners(summaries),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--cache-ttl-minutes", type=int, default=DEFAULT_CACHE_TTL_MINUTES)
    parser.add_argument(
        "--cache-write-multiplier",
        type=float,
        default=1.25,
        help="Multiplier over base input price used for cache writes/store cost.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes to use. Higher values use more memory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / CHECKPOINT_DIRNAME,
        help="Directory for resumable per-session checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger("headroom.transforms").setLevel(logging.WARNING)
    logging.getLogger("headroom.proxy").setLevel(logging.WARNING)
    session_files = select_session_files(args.root, max_sessions=args.max_sessions)
    if not session_files:
        print(f"No Claude session replays found under {args.root}")
        return 1
    dataset, observed = build_dataset_and_observed_from_files(
        session_files,
        cache_write_multiplier=args.cache_write_multiplier,
    )
    print(
        f"[load] loaded {dataset.sessions} sessions from {args.root}"
        + (f" (max_sessions={args.max_sessions})" if args.max_sessions is not None else ""),
        flush=True,
    )
    summaries = simulate_session_files(
        session_files,
        dataset,
        cache_ttl_minutes=args.cache_ttl_minutes,
        cache_write_multiplier=args.cache_write_multiplier,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
    )
    md_path, json_path, html_path = write_report(args.output_dir, dataset, observed, summaries)
    print_observed_console_report(observed)
    print_console_report(dataset, summaries)
    print()
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
