# Dashboard

The Headroom proxy serves a local dashboard at:

```bash
http://localhost:8787/dashboard
```

Start the proxy first:

```bash
headroom proxy
```

Then open the dashboard in your browser:

```bash
open http://localhost:8787/dashboard
```

On Windows:

```powershell
start http://localhost:8787/dashboard
```

## What it shows

The dashboard is the human-facing view for proxy observability. It shows:

- live proxy health
- current-session requests, tokens, cache hits, failures, and savings
- durable lifetime savings loaded from local state
- hourly, daily, weekly, and monthly savings history
- recent transformed/compressed request events
- export buttons for savings history

It is served by the same proxy process and does not require a separate web
server.

## Data sources

The dashboard reads the same HTTP endpoints you can query directly:

| Endpoint | Used for |
|---|---|
| `GET /health` | proxy readiness and runtime status |
| `GET /stats?cached=1` | short-TTL dashboard stats snapshot |
| `GET /stats-history` | durable history and chart rollups |
| `GET /transformations/feed?limit=50` | recent compression/transformation events |

If the dashboard looks empty, check these endpoints first:

```bash
curl http://localhost:8787/health
curl http://localhost:8787/stats
curl http://localhost:8787/stats-history
curl "http://localhost:8787/transformations/feed?limit=5"
```

## Durable history

Savings history is stored locally and survives proxy restarts. By default, the
proxy stores it under the Headroom workspace state directory:

```text
~/.headroom/proxy_savings.json
```

You can override the file directly:

```bash
HEADROOM_SAVINGS_PATH=/path/to/proxy_savings.json headroom proxy
```

Or relocate the whole state root:

```bash
HEADROOM_WORKSPACE_DIR=/path/to/headroom-state headroom proxy
```

See [Filesystem Contract](filesystem-contract.md) for the full path rules.

## Export history

The dashboard export buttons use `/stats-history` under the hood. The same data
is available from the CLI:

```bash
curl "http://localhost:8787/stats-history?format=csv&series=daily"
curl "http://localhost:8787/stats-history?format=csv&series=weekly"
curl "http://localhost:8787/stats-history?format=csv&series=monthly"
curl "http://localhost:8787/stats-history?history_mode=full"
```

## Persistent proxy

For an always-on proxy that keeps the dashboard available across terminal
sessions, install a persistent service:

```bash
headroom install apply --preset persistent-service --providers auto
```

See [Persistent Installs](persistent-installs.md).
