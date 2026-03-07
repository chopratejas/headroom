"""Analyze headroom proxy logs for performance insights.

Parses PERF log lines from ~/.headroom/logs/proxy.log* and produces
actionable reports on token savings, cache efficiency, and transform impact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LOG_DIR = Path.home() / ".headroom" / "logs"

# Matches: 2026-03-07 13:38:31,009 - headroom.proxy - INFO - [hr_...] PERF model=... ...
_PERF_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .* \[(?P<rid>[^\]]+)\] PERF (?P<kv>.+)$"
)

# Matches: content_router: 51 msgs — ...
_ROUTER_RE = re.compile(r"content_router: (?P<msgs>\d+) msgs — (?P<detail>.+)$")

# Matches: Transform content_router: 52503 -> 26006 tokens (saved 26497)
_TRANSFORM_RE = re.compile(
    r"Transform (?P<name>\w+): (?P<before>\d+) -> (?P<after>\d+) tokens \(saved (?P<saved>\d+)\)"
)

# Matches: Pipeline complete: 52503 -> 26006 tokens (saved 26497, 50.5% reduction)
_PIPELINE_RE = re.compile(
    r"Pipeline complete: (?P<before>\d+) -> (?P<after>\d+) tokens "
    r"\(saved (?P<saved>\d+), (?P<pct>[\d.]+)% reduction\)"
)

# Matches: TOIN: 105 patterns, 3837 compressions, 0 retrievals, 0.0% retrieval rate
_TOIN_RE = re.compile(
    r"TOIN: (?P<patterns>\d+) patterns, (?P<compressions>\d+) compressions, "
    r"(?P<retrievals>\d+) retrievals, (?P<rate>[\d.]+)% retrieval rate"
)


def _parse_kv(kv_str: str) -> dict[str, str]:
    """Parse key=value pairs from a PERF log line."""
    result: dict[str, str] = {}
    for part in kv_str.split():
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


@dataclass
class PerfRecord:
    """A single parsed PERF log entry."""

    timestamp: str
    request_id: str
    model: str = ""
    num_messages: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_hit_pct: int = 0
    optimization_ms: float = 0
    transforms: list[str] = field(default_factory=list)


@dataclass
class RouterRecord:
    """A parsed content_router summary line."""

    timestamp: str
    num_messages: int = 0
    compressed: int = 0
    excluded: int = 0
    skipped: int = 0
    unchanged: int = 0
    content_blocks: int = 0


@dataclass
class TransformRecord:
    """A parsed per-transform line."""

    timestamp: str
    name: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0


@dataclass
class ToinRecord:
    """A parsed TOIN status line."""

    timestamp: str
    patterns: int = 0
    compressions: int = 0
    retrievals: int = 0
    retrieval_rate: float = 0.0


@dataclass
class PerfReport:
    """Aggregated performance report."""

    perf_records: list[PerfRecord] = field(default_factory=list)
    router_records: list[RouterRecord] = field(default_factory=list)
    transform_records: list[TransformRecord] = field(default_factory=list)
    toin_records: list[ToinRecord] = field(default_factory=list)
    log_files_read: int = 0
    total_lines_parsed: int = 0


def parse_log_files(last_n_hours: float = 168.0) -> PerfReport:
    """Parse all proxy log files and return structured records.

    Args:
        last_n_hours: Only include records from the last N hours (default 7 days).

    Returns:
        PerfReport with all parsed records.
    """
    report = PerfReport()

    if not LOG_DIR.exists():
        return report

    # Collect log files: proxy.log, proxy.log.1, proxy.log.2, ...
    log_files = sorted(LOG_DIR.glob("proxy.log*"), key=lambda p: p.stat().st_mtime)

    for log_file in log_files:
        report.log_files_read += 1
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    report.total_lines_parsed += 1
                    line = line.rstrip()

                    # PERF lines (richest data)
                    m = _PERF_RE.match(line)
                    if m:
                        kv = _parse_kv(m.group("kv"))
                        transforms_str = kv.get("transforms", "none")
                        transforms = transforms_str.split(",") if transforms_str != "none" else []
                        report.perf_records.append(
                            PerfRecord(
                                timestamp=m.group("ts"),
                                request_id=m.group("rid"),
                                model=kv.get("model", ""),
                                num_messages=int(kv.get("msgs", 0)),
                                tokens_before=int(kv.get("tok_before", 0)),
                                tokens_after=int(kv.get("tok_after", 0)),
                                tokens_saved=int(kv.get("tok_saved", 0)),
                                cache_read=int(kv.get("cache_read", 0)),
                                cache_write=int(kv.get("cache_write", 0)),
                                cache_hit_pct=int(kv.get("cache_hit_pct", 0)),
                                optimization_ms=float(kv.get("opt_ms", 0)),
                                transforms=transforms,
                            )
                        )
                        continue

                    # content_router summary lines
                    if "content_router:" in line and "msgs" in line:
                        m2 = _ROUTER_RE.search(line)
                        if m2:
                            ts = line[:23]
                            detail = m2.group("detail")
                            rec = RouterRecord(
                                timestamp=ts,
                                num_messages=int(m2.group("msgs")),
                            )
                            # Parse counts from detail string
                            for part in detail.split(","):
                                part = part.strip()
                                num_match = re.match(r"(\d+)\s+(\w+)", part)
                                if num_match:
                                    count = int(num_match.group(1))
                                    kind = num_match.group(2)
                                    if kind == "compressed":
                                        rec.compressed = count
                                    elif kind == "excluded":
                                        rec.excluded = count
                                    elif kind == "skipped":
                                        rec.skipped = count
                                    elif kind == "unchanged":
                                        rec.unchanged = count
                                    elif kind == "content" and "block" in part:
                                        rec.content_blocks = count
                            report.router_records.append(rec)
                            continue

                    # Per-transform lines
                    m3 = _TRANSFORM_RE.search(line)
                    if m3:
                        ts = line[:23]
                        report.transform_records.append(
                            TransformRecord(
                                timestamp=ts,
                                name=m3.group("name"),
                                tokens_before=int(m3.group("before")),
                                tokens_after=int(m3.group("after")),
                                tokens_saved=int(m3.group("saved")),
                            )
                        )
                        continue

                    # TOIN status lines
                    m4 = _TOIN_RE.search(line)
                    if m4:
                        ts = line[:23]
                        report.toin_records.append(
                            ToinRecord(
                                timestamp=ts,
                                patterns=int(m4.group("patterns")),
                                compressions=int(m4.group("compressions")),
                                retrievals=int(m4.group("retrievals")),
                                retrieval_rate=float(m4.group("rate")),
                            )
                        )

        except OSError:
            continue

    return report


def format_report(report: PerfReport) -> str:
    """Format a PerfReport into a human-readable string."""
    lines: list[str] = []

    if not report.perf_records and not report.router_records:
        lines.append("No performance data found in ~/.headroom/logs/")
        lines.append("")
        lines.append("Start the proxy to begin collecting data:")
        lines.append("  headroom proxy")
        return "\n".join(lines)

    # Header
    lines.append("Headroom Performance Report")
    lines.append("=" * 60)
    lines.append("")

    records = report.perf_records

    if records:
        # Overview
        total_before = sum(r.tokens_before for r in records)
        total_after = sum(r.tokens_after for r in records)
        total_saved = sum(r.tokens_saved for r in records)
        pct = (total_saved / total_before * 100) if total_before > 0 else 0

        models = {r.model for r in records}
        lines.append(f"Requests:     {len(records)}")
        lines.append(f"Models:       {', '.join(sorted(models))}")
        lines.append(
            f"Tokens:       {total_before:,} input -> {total_after:,} after transforms "
            f"({pct:.1f}% reduction)"
        )
        lines.append(f"Total saved:  {total_saved:,} tokens")
        lines.append("")

        # Cache analysis
        cache_records = [r for r in records if (r.cache_read + r.cache_write) > 0]
        if cache_records:
            lines.append("Cache Performance")
            lines.append("-" * 40)
            total_cr = sum(r.cache_read for r in cache_records)
            total_cw = sum(r.cache_write for r in cache_records)
            total_cache = total_cr + total_cw
            hit_pct = (total_cr / total_cache * 100) if total_cache > 0 else 0
            lines.append(f"  Cache read:    {total_cr:,} tokens")
            lines.append(f"  Cache write:   {total_cw:,} tokens")
            lines.append(f"  Hit rate:      {hit_pct:.1f}%")

            # Identify cache instability: requests where write >> read
            unstable = [r for r in cache_records if r.cache_write > r.cache_read * 2]
            if unstable:
                lines.append(
                    f"  Unstable:      {len(unstable)}/{len(cache_records)} requests "
                    f"had cache_write > 2x cache_read"
                )

            # Show cache progression (first 5 vs last 5)
            if len(cache_records) >= 10:
                first5_cr = sum(r.cache_read for r in cache_records[:5])
                first5_cw = sum(r.cache_write for r in cache_records[:5])
                last5_cr = sum(r.cache_read for r in cache_records[-5:])
                last5_cw = sum(r.cache_write for r in cache_records[-5:])
                lines.append(f"  First 5 avg:   read={first5_cr // 5:,} write={first5_cw // 5:,}")
                lines.append(f"  Last 5 avg:    read={last5_cr // 5:,} write={last5_cw // 5:,}")
                if last5_cr > first5_cr * 2:
                    lines.append("  -> Cache stabilizing over conversation lifetime")
                elif first5_cw > first5_cr * 3:
                    lines.append(
                        "  ! Early turns have poor cache hits — "
                        "compression decisions may be flipping"
                    )
            lines.append("")

        # Optimization latency
        opt_times = [r.optimization_ms for r in records if r.optimization_ms > 0]
        if opt_times:
            avg_opt = sum(opt_times) / len(opt_times)
            max_opt = max(opt_times)
            lines.append("Optimization Overhead")
            lines.append("-" * 40)
            lines.append(f"  Average:  {avg_opt:.0f}ms")
            lines.append(f"  Max:      {max_opt:.0f}ms")
            slow = [t for t in opt_times if t > 500]
            if slow:
                lines.append(f"  >500ms:   {len(slow)} requests")
            lines.append("")

        # Conversation size distribution
        msg_counts = [r.num_messages for r in records if r.num_messages > 0]
        if msg_counts:
            lines.append("Conversation Size")
            lines.append("-" * 40)
            lines.append(f"  Min msgs:  {min(msg_counts)}")
            lines.append(f"  Max msgs:  {max(msg_counts)}")
            lines.append(f"  Avg msgs:  {sum(msg_counts) // len(msg_counts)}")
            lines.append("")

    # Transform effectiveness (from transform_records)
    if report.transform_records:
        lines.append("Transform Effectiveness")
        lines.append("-" * 40)
        by_name: dict[str, list[TransformRecord]] = {}
        for tr in report.transform_records:
            by_name.setdefault(tr.name, []).append(tr)
        for name, recs in sorted(by_name.items(), key=lambda x: -sum(r.tokens_saved for r in x[1])):
            total_s = sum(r.tokens_saved for r in recs)
            total_b = sum(r.tokens_before for r in recs)
            avg_pct = (total_s / total_b * 100) if total_b > 0 else 0
            lines.append(
                f"  {name}: {avg_pct:.1f}% avg reduction, {len(recs)} uses, {total_s:,} saved"
            )
        lines.append("")

    # Router routing breakdown
    if report.router_records:
        lines.append("Content Router Routing")
        lines.append("-" * 40)
        total_compressed = sum(r.compressed for r in report.router_records)
        total_excluded = sum(r.excluded for r in report.router_records)
        total_skipped = sum(r.skipped for r in report.router_records)
        total_unchanged = sum(r.unchanged for r in report.router_records)
        total_all = total_compressed + total_excluded + total_skipped + total_unchanged
        if total_all > 0:
            lines.append(
                f"  Compressed:  {total_compressed} ({total_compressed / total_all * 100:.0f}%)"
            )
            lines.append(
                f"  Excluded:    {total_excluded} ({total_excluded / total_all * 100:.0f}%) — Read/Glob outputs"
            )
            lines.append(
                f"  Skipped:     {total_skipped} ({total_skipped / total_all * 100:.0f}%) — <50 words"
            )
            lines.append(
                f"  Unchanged:   {total_unchanged} ({total_unchanged / total_all * 100:.0f}%) — ratio too high"
            )
        if total_excluded > total_compressed * 3:
            lines.append("  ! Excluded tools dominate — consider compressing stale Read outputs")
        lines.append("")

    # TOIN status
    if report.toin_records:
        latest = report.toin_records[-1]
        lines.append("TOIN Learning")
        lines.append("-" * 40)
        lines.append(f"  Patterns:     {latest.patterns}")
        lines.append(f"  Compressions: {latest.compressions:,}")
        lines.append(f"  Retrievals:   {latest.retrievals} ({latest.retrieval_rate}%)")
        if latest.retrieval_rate == 0 and latest.compressions > 100:
            lines.append("  ! 0% retrieval rate — TOIN learning but never used")
        lines.append("")

    # Recommendations
    recommendations = _generate_recommendations(report)
    if recommendations:
        lines.append("Recommendations")
        lines.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    # Footer
    lines.append(
        f"Log files: {report.log_files_read} | Lines parsed: {report.total_lines_parsed:,}"
    )
    lines.append(f"Log dir: {LOG_DIR}")

    return "\n".join(lines)


def _generate_recommendations(report: PerfReport) -> list[str]:
    """Generate actionable recommendations from the report data."""
    recs: list[str] = []

    if report.perf_records:
        cache_recs = [r for r in report.perf_records if (r.cache_read + r.cache_write) > 0]
        if cache_recs:
            total_cr = sum(r.cache_read for r in cache_recs)
            total_cw = sum(r.cache_write for r in cache_recs)
            if total_cw > total_cr * 1.5:
                recs.append(
                    "Cache prefix unstable — compression decisions may be flipping "
                    "across turns due to adaptive min_ratio threshold"
                )

            # Check early-turn instability
            if len(cache_recs) >= 5:
                first5 = cache_recs[:5]
                early_ratio = sum(r.cache_read for r in first5) / max(
                    1, sum(r.cache_write for r in first5)
                )
                if early_ratio < 0.5:
                    recs.append(
                        "First 5 turns have very low cache hit ratio — "
                        "consider pinning compression decisions for prefix stability"
                    )

        # Optimization latency
        slow = [r for r in report.perf_records if r.optimization_ms > 500]
        if len(slow) > len(report.perf_records) * 0.2:
            recs.append(
                f"{len(slow)} requests took >500ms for optimization — "
                "consider disabling LLMLingua or reducing transform pipeline"
            )

    if report.router_records:
        total_excluded = sum(r.excluded for r in report.router_records)
        total_compressed = sum(r.compressed for r in report.router_records)
        if total_excluded > 0 and total_compressed > 0:
            if total_excluded > total_compressed * 3:
                recs.append(
                    "Read/Glob outputs are majority of messages but excluded — "
                    "compress stale reads (>10 turns old) for significant savings"
                )

    if report.toin_records:
        latest = report.toin_records[-1]
        if latest.retrieval_rate == 0 and latest.compressions > 100:
            recs.append(
                "TOIN has 0% retrieval rate with "
                f"{latest.compressions:,} compressions — review CCR integration"
            )

    # Check cache_aligner effectiveness from transform records
    for tr in report.transform_records:
        if tr.name == "cache_aligner" and tr.tokens_saved < 10:
            recs.append(
                "cache_aligner saving <10 tokens — "
                "consider disabling (system prompt likely has no dynamic content)"
            )
            break

    return recs
