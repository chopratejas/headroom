# Headroom Bedrock Proxy — CLI Setup (no Docker)

Run the Headroom compression proxy in front of AWS Bedrock from the CLI —
the non-Docker equivalent of `docker-compose.bedrock.yml`. Claude Code points
at the local proxy, the proxy translates + SigV4-signs to Bedrock via its
LiteLLM backend, and your traffic gets compressed on the way through.

> Uses the Python `headroom proxy --backend bedrock` path (LiteLLM converse +
> SigV4). This is distinct from the native Rust `headroom-proxy` binary
> documented in [`bedrock.md`](./bedrock.md).

## 1. Prerequisites (install once)

```sh
# Rust toolchain — headroom compiles a native extension (headroom/_core.so) on install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# uv (Python project/runtime manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# AWS CLI v2 — for credentials (skip if already installed)
```

## 2. Clone the repo and check out the branch

```sh
git clone https://github.com/KennethWKZ/headroom.git
cd headroom
git checkout feat/bedrock-inference-profile-arn
```

## 3. Install headroom with Bedrock support

```sh
uv tool install ".[proxy,code,bedrock]"
headroom --version   # verify it is on PATH
```

The `bedrock` extra pulls in `boto3` and `botocore[crt]` (the latter is required
for the `aws login` / SSO credential providers).

## 4. Get AWS credentials working

Use whatever your org uses, then confirm before starting the proxy:

```sh
aws login --profile default       # or: aws sso login --profile default
aws sts get-caller-identity        # must return your account — the proxy needs this
```

The proxy reads `~/.aws` directly (no mount needed — that is a Docker-only concern).

## 5. Start the Bedrock proxy

```sh
headroom proxy \
  --backend bedrock \
  --region ap-southeast-1 \
  --bedrock-profile default \
  --memory \
  --port 8789
```

Leave it running. By default it binds `127.0.0.1` — fine for local dev. Add
`--host 0.0.0.0` only if another machine must reach it.

## 6. Point Claude Code at the proxy

Set these in your client (e.g. a ccs `bedrock.settings.json`, or plain env):

```sh
ANTHROPIC_BASE_URL=http://localhost:8789
ANTHROPIC_MODEL=arn:aws:bedrock:ap-southeast-1:<ACCOUNT_ID>:application-inference-profile/<PROFILE_ID>[1m]
```

- The model id may be a short Claude id (`claude-sonnet-4-...`), a
  region/global inference profile, or a full inference-profile ARN.
- Append `[1m]` to request the 1M context window (forwarded as the
  `context-1m-2025-08-07` beta).
- **Do NOT set `CLAUDE_CODE_USE_BEDROCK`.** That makes Claude Code talk to AWS
  directly, bypassing the proxy — nothing gets compressed.

Then launch Claude Code as usual.

## CLI ↔ docker-compose.bedrock.yml mapping

| `docker-compose.bedrock.yml` | CLI equivalent |
|---|---|
| `command: --host 0.0.0.0 --memory --backend bedrock` | `--memory --backend bedrock` |
| `HEADROOM_BEDROCK_REGION=${AWS_REGION}` | `--region ap-southeast-1` |
| `AWS_PROFILE=default` | `--bedrock-profile default` |
| `${HOME}/.aws:/home/nonroot/.aws:ro` mount | native — reads `~/.aws` directly |
| port `8789:8787` | `--port 8789` |

The CLI run is simpler than the container: no UID/HOME juggling and no bind
mount, because the proxy runs as your own user and reads `~/.aws` in place.

## Updating later

A plain (non-editable) `uv tool install` is a snapshot. After `git pull`,
reinstall to pick up changes:

```sh
git pull
uv tool install --reinstall ".[proxy,code,bedrock]"
```

To develop against the source live instead, install editable
(`uv tool install --editable ".[proxy,code,bedrock]"`) — then edits load without
a reinstall.
