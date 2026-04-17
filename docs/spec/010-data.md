# 010. Data

**Status:** done

## Storage Overview

| Store | Location | Type | Purpose |
|-------|----------|------|---------|
| SQLite | `headroom.db` (cwd or `~/.headroom/`) | Local file | Metrics, sessions, compression store |
| Memory DB | `~/.headroom/headroom_memory.db` | Local file | Memory system (optional) |
| Vector Index | `~/.headroom/headroom_memory_vectors.db` | Local file | Semantic search (optional, requires sqlite-vec) |
| Graph Store | `~/.headroom/headroom_memory_graph.db` | Local file | Memory relationships |
| Compression cache | `~/.headroom/cache/` | Directory | Semantic + summary cache |

**Note:** Default store URL is `sqlite:///headroom.db` (relative to working directory).

---

## SQLite Schema

### Core Tables

**sessions:**
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON
);
```

**requests:**
```sql
CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    stream INTEGER NOT NULL,
    mode TEXT NOT NULL,
    tokens_input_before INTEGER NOT NULL,
    tokens_input_after INTEGER NOT NULL,
    tokens_output INTEGER,
    block_breakdown TEXT NOT NULL,  -- JSON
    waste_signals TEXT NOT NULL,  -- JSON
    stable_prefix_hash TEXT,
    cache_alignment_score REAL,
    cached_tokens INTEGER,
    transforms_applied TEXT NOT NULL,  -- JSON
    tool_units_dropped INTEGER DEFAULT 0,
    turns_dropped INTEGER DEFAULT 0,
    messages_hash TEXT,
    error TEXT
);

CREATE INDEX idx_timestamp ON requests(timestamp);
CREATE INDEX idx_model ON requests(model);
CREATE INDEX idx_mode ON requests(mode);
```

**cache_entries:**
```sql
CREATE TABLE cache_entries (
    cache_key TEXT PRIMARY KEY,
    input_hash TEXT NOT NULL,
    output TEXT NOT NULL,
    compression_type TEXT NOT NULL,
    tokens_before INTEGER NOT NULL,
    tokens_after INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl INTEGER DEFAULT 3600,
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**compression_store (CCR):**
```sql
CREATE TABLE compression_store (
    hash TEXT PRIMARY KEY,
    original TEXT NOT NULL,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);
```

---

## Environment Variables

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `HEADROOM_DB_PATH` | `~/.headroom/` | Base path for databases |
| `HEADROOM_DB_URL` | - | External database URL (overrides default) |
| `HEADROOM_CACHE_PATH` | `~/.headroom/cache/` | Compression cache location |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HEADROOM_CACHE_ENABLED` | `true` | Enable compression cache |
| `HEADROOM_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `HEADROOM_CACHE_MAX_SIZE` | `10000` | Max cache entries |

---

## Data Retention

| Data Type | Retention | Auto-cleanup |
|-----------|-----------|:------------:|
| Savings history | Forever | No |
| Session history | 30 days | Yes (configurable) |
| Compression cache | 7 days | Yes |
| Telemetry | 90 days | Yes |
| Dashboard state | 30 days | Yes |

---

## Data Export

### Savings Export

```bash
curl http://localhost:8787/stats
```

Response:
```json
{
  "savings": [
    {
      "date": "2026-04-16",
      "original_tokens": 8192,
      "compressed_tokens": 5325,
      "savings_percentage": 0.35
    }
  ],
  "total_savings": 1234567
}
```

### Session Export

```bash
curl http://localhost:8787/stats
```

---

## Backup

### Manual Backup

```bash
tar -czf headroom-backup.tar.gz ~/.headroom/
```

### External Database

Set `HEADROOM_DB_URL` for external storage:

```bash
export HEADROOM_DB_URL=postgresql://user:pass@host:5432/headroom
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial data document |
