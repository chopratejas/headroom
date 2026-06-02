```
  ██╗  ██╗███████╗ █████╗ ██████╗ ██████╗  ██████╗  ██████╗ ███╗   ███╗
  ██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗████╗ ████║
  ███████║█████╗  ███████║██║  ██║██████╔╝██║   ██║██║   ██║██╔████╔██║
  ██╔══██║██╔══╝  ██╔══██║██║  ██║██╔══██╗██║   ██║██║   ██║██║╚██╔╝██║
  ██║  ██║███████╗██║  ██║██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
                  面向 AI Agent 的上下文压缩层
```

<p align="center"><strong>节省 60–95% Token · 库 · 代理 · MCP · 6 种算法 · 本地优先 · 可逆</strong></p>

<p align="center">
  <a href="https://github.com/chopratejas/headroom/actions/workflows/ci.yml"><img src="https://github.com/chopratejas/headroom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://app.codecov.io/gh/chopratejas/headroom"><img src="https://codecov.io/gh/chopratejas/headroom/graph/badge.svg" alt="codecov"></a>
  <a href="https://pypi.org/project/headroom-ai/"><img src="https://img.shields.io/pypi/v/headroom-ai.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/headroom-ai"><img src="https://img.shields.io/npm/v/headroom-ai.svg" alt="npm"></a>
  <a href="https://huggingface.co/chopratejas/kompress-base"><img src="https://img.shields.io/badge/model-Kompress--base-yellow.svg" alt="Model: Kompress-base"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://headroom-docs.vercel.app/docs"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Docs"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://headroom-docs.vercel.app/docs">文档</a> ·
  <a href="#60-秒上手">安装</a> ·
  <a href="#实测效果">实测</a> ·
  <a href="#agent-兼容矩阵">支持的 Agent</a> ·
  <a href="https://discord.gg/yRmaUNpsPJ">Discord</a> ·
  <a href="llms.txt">llms.txt</a>
</p>

<p align="center"><sub>
  <b>AI Agent / LLM 请读这里：</b>本仓库的 <a href="llms.txt"><code>/llms.txt</code></a>，或在线版 <a href="https://headroom-docs.vercel.app/llms.txt">索引</a> / <a href="https://headroom-docs.vercel.app/llms-full.txt">完整文档</a>。
</sub></p>

---

> Headroom 会在 AI Agent 把内容送给 LLM 之前，先压缩它读到的一切——工具输出、日志、RAG 切片、文件、对话历史。答案不变，Token 只剩一小部分。

<p align="center">
  <img src="HeadroomDemo-Fast.gif" alt="Headroom 实战演示" width="820">
  <br/><sub>真实样例：10,144 → 1,260 tokens，依然找出了同一条 FATAL 日志。</sub>
</p>

## 功能一览

- **库** —— Python 或 TypeScript 里直接 `compress(messages)`，随处可用
- **代理** —— `headroom proxy --port 8787`，无需改一行代码，任意语言均可接入
- **Agent 包装** —— `headroom wrap claude|codex|cursor|aider|copilot`，一条命令搞定
- **MCP 服务** —— 任何 MCP 客户端都可调用 `headroom_compress`、`headroom_retrieve`、`headroom_stats`
- **跨 Agent 记忆** —— Claude / Codex / Gemini 共享同一份存储，自动去重
- **`headroom learn`** —— 从失败会话里挖出规律，自动写回 `CLAUDE.md` / `AGENTS.md`
- **可逆压缩 (CCR)** —— 原文从不丢弃；LLM 需要时按需取回

## 工作原理（30 秒读完）

```
 你的 Agent / 应用
   (Claude Code、Cursor、Codex、LangChain、Agno、Strands、自研代码……)
        │   prompt · 工具输出 · 日志 · RAG 结果 · 文件
        ▼
    ┌────────────────────────────────────────────────────┐
    │  Headroom   (在本地运行 —— 数据不出本机)            │
    │  ────────────────────────────────────────────────  │
    │  CacheAligner  →  ContentRouter  →  CCR            │
    │                    ├─ SmartCrusher   (JSON)        │
    │                    ├─ CodeCompressor (AST)         │
    │                    └─ Kompress-base  (text, HF)    │
    │                                                    │
    │  跨 Agent 记忆  ·  headroom learn  ·  MCP          │
    └────────────────────────────────────────────────────┘
        │   压缩后的 prompt  +  retrieval 工具
        ▼
 LLM 服务  (Anthropic · OpenAI · Bedrock · …)
```

- **ContentRouter** —— 识别内容类型，挑选合适的压缩器
- **SmartCrusher / CodeCompressor / Kompress-base** —— 分别处理 JSON、AST、自然语言文本
- **CacheAligner** —— 稳定前缀，让 LLM 服务商的 KV 缓存真正命中
- **CCR** —— 原文落在本地；LLM 按需调用 `headroom_retrieve` 取回

→ [架构说明](https://headroom-docs.vercel.app/docs/architecture) · [CCR 可逆压缩](https://headroom-docs.vercel.app/docs/ccr) · [Kompress-base 模型卡](https://huggingface.co/chopratejas/kompress-base)

## 60 秒上手

```bash
# 1 —— 安装
pip install "headroom-ai[all]"          # Python
npm install headroom-ai                 # Node / TypeScript

# 2 —— 选择使用方式
headroom wrap claude                    # 包装一个编码 Agent
headroom proxy --port 8787              # 直接挂代理，零代码改动
# 或：from headroom import compress      # 当成库内嵌使用

# 3 —— 查看节省效果
headroom stats
```

细粒度安装选项：`[proxy]`、`[mcp]`、`[ml]`、`[agno]`、`[langchain]`、`[evals]`。需要 **Python 3.10+**。

## 实测效果

**在真实 Agent 工作负载下的节省：**

| 工作负载                      |  压缩前 |  压缩后 |    节省 |
|-------------------------------|-------:|-------:|--------:|
| 代码搜索（100 条结果）          | 17,765 |  1,408 | **92%** |
| SRE 故障排查                   | 65,694 |  5,118 | **92%** |
| GitHub Issue 分类              | 54,174 | 14,761 | **73%** |
| 代码库探索                     | 78,502 | 41,254 | **47%** |

**在标准基准上准确率保持不变：**

| 基准         | 类别  | N   | 基线值 | Headroom | 差值           |
|--------------|------|----:|------:|---------:|---------------|
| GSM8K        | 数学  | 100 | 0.870 |    0.870 | **±0.000**    |
| TruthfulQA   | 事实  | 100 | 0.530 |    0.560 | **+0.030**    |
| SQuAD v2     | QA   | 100 |    —  |  **97%** | 压缩 19%       |
| BFCL         | 工具  | 100 |    —  |  **97%** | 压缩 32%       |

复现命令：`python -m headroom.evals suite --tier 1` · [完整基准与方法学](https://headroom-docs.vercel.app/docs/benchmarks)

## Agent 兼容矩阵

| Agent       | `headroom wrap` | 备注                              |
|-------------|:---------------:|----------------------------------|
| Claude Code | ●               | `--memory` · `--code-graph`      |
| Codex       | ●               | 与 Claude 共享记忆                |
| Cursor      | ●               | 打印配置 —— 粘贴一次即可          |
| Aider       | ●               | 启动代理并拉起 Aider              |
| Copilot CLI | ●               | 启动代理并拉起 Copilot            |
| OpenClaw    | ●               | 作为 ContextEngine 插件安装       |

任何 OpenAI 兼容的客户端都可以通过 `headroom proxy` 接入。MCP 原生支持：`headroom mcp install`。

## 适用 / 不适用场景

**适合你，如果……**
- 你每天都在跑 AI 编码 Agent，希望在不改代码的前提下省 Token
- 你同时使用多个 Agent，需要它们共享一份记忆
- 你需要可逆压缩 —— 原文随时可通过 CCR 取回

**别用 Headroom，如果……**
- 你只用单一服务商的原生上下文压缩，也不需要跨 Agent 记忆
- 你在沙盒环境里，本地无法跑后台进程

<details>
<summary><b>集成 —— 把 Headroom 装进任意技术栈</b></summary>

| 你的环境               | 接入方式                                                          |
|------------------------|------------------------------------------------------------------|
| 任意 Python 应用        | `compress(messages, model=…)`                                    |
| 任意 TypeScript 应用    | `await compress(messages, { model })`                            |
| Anthropic / OpenAI SDK | `withHeadroom(new Anthropic())` · `withHeadroom(new OpenAI())`   |
| Vercel AI SDK          | `wrapLanguageModel({ model, middleware: headroomMiddleware() })` |
| LiteLLM                | `litellm.callbacks = [HeadroomCallback()]`                       |
| LangChain              | `HeadroomChatModel(your_llm)`                                    |
| Agno                   | `HeadroomAgnoModel(your_model)`                                  |
| Strands                | [Strands 指南](https://headroom-docs.vercel.app/docs/strands)    |
| ASGI 应用              | `app.add_middleware(CompressionMiddleware)`                      |
| 多 Agent               | `SharedContext().put / .get`                                     |
| MCP 客户端             | `headroom mcp install`                                           |

</details>

<details>
<summary><b>内部构件</b></summary>

- **SmartCrusher** —— 通用 JSON 压缩：字典数组、嵌套对象、混合类型都能处理。
- **CodeCompressor** —— 基于 AST，支持 Python、JS、Go、Rust、Java、C++。
- **Kompress-base** —— 在 HuggingFace 上开源的模型，使用 Agent 轨迹训练。
- **图像压缩** —— 通过训练好的 ML 路由器，体积可缩减 40–90%。
- **CacheAligner** —— 稳定前缀，让 Anthropic / OpenAI 的 KV 缓存真的能命中。
- **IntelligentContext** —— 基于打分的上下文裁剪，结合学习到的重要性权重。
- **CCR** —— 可逆压缩；LLM 需要时随时取回原文。
- **跨 Agent 记忆** —— 共享存储、来源追踪、自动去重。
- **SharedContext** —— 在多 Agent 工作流之间传递已压缩的上下文。
- **`headroom learn`** —— 基于插件的失败挖掘机制，支持 Claude、Codex、Gemini。

</details>

<details>
<summary><b>流水线内部</b></summary>

Headroom 通过 `compress()`、SDK 与代理三种入口共享同一套稳定的请求生命周期：

`Setup` → `Pre-Start` → `Post-Start` → `Input Received` → `Input Cached` → `Input Routed` → `Input Compressed` → `Input Remembered` → `Pre-Send` → `Post-Send` → `Response Received`

- **Transform（变换器）** 负责实际工作：CacheAligner、ContentRouter、SmartCrusher、CodeCompressor、Kompress-base、IntelligentContext / RollingWindow。
- **流水线扩展** 通过 `on_pipeline_event(...)` 观察或定制生命周期。
- **压缩钩子** 与生命周期并行，作为额外的扩展挂载点。
- **代理扩展** 是服务端 / 应用层的集成点，覆盖 ASGI 中间件、路由与启动策略。

服务商与具体工具相关的逻辑放在 `headroom/providers/` 下，让核心保持专注于生命周期、调度和策略。

- **CLI / 工具切片**：`headroom/providers/claude`、`copilot`、`codex`、`openclaw`
- **服务商运行时切片**：`headroom/providers/claude`、`gemini`，加上 `headroom/providers/registry.py` 中的共享后端 / 运行时分发
- **核心文件保持编排优先**：`wrap.py`、`client.py`、`cli/proxy.py`、`proxy/server.py` 把环境塑形、API 目标归一、后端选择、传输分发都委托给服务商专属代码。

</details>

## 安装

```bash
pip install "headroom-ai[all]"          # Python，完整功能
npm install headroom-ai                 # TypeScript / Node
docker pull ghcr.io/chopratejas/headroom:latest
```

细粒度选项：`[proxy]`、`[mcp]`、`[ml]`（Kompress-base）、`[agno]`、`[langchain]`、`[evals]`。需要 **Python 3.10+**。

使用 `pipx`？请显式指定支持的解释器版本：

```bash
pipx install --python python3.13 "headroom-ai[all]"
```

→ [完整安装指南](https://headroom-docs.vercel.app/docs/installation) —— Docker 镜像标签、常驻服务、PowerShell、Devcontainer。

## headroom learn

<p align="center">
  <img src="headroom_learn.gif" alt="headroom learn 演示" width="720">
</p>

`headroom learn` —— 挖掘失败会话，把改进建议写回 `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`。

## 文档

| 入门指南                                                                       | 深入阅读                                                                            |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [快速上手](https://headroom-docs.vercel.app/docs/quickstart)                   | [架构设计](https://headroom-docs.vercel.app/docs/architecture)                      |
| [代理模式](https://headroom-docs.vercel.app/docs/proxy)                        | [压缩原理](https://headroom-docs.vercel.app/docs/how-compression-works)             |
| [MCP 工具](https://headroom-docs.vercel.app/docs/mcp)                          | [CCR —— 可逆压缩](https://headroom-docs.vercel.app/docs/ccr)                       |
| [跨 Agent 记忆](https://headroom-docs.vercel.app/docs/memory)                  | [缓存优化](https://headroom-docs.vercel.app/docs/cache-optimization)                |
| [失败学习](https://headroom-docs.vercel.app/docs/failure-learning)             | [基准测试](https://headroom-docs.vercel.app/docs/benchmarks)                       |
| [配置说明](https://headroom-docs.vercel.app/docs/configuration)                | [已知局限](https://headroom-docs.vercel.app/docs/limitations)                      |

## 横向对比

Headroom 在 **本地** 运行，覆盖 **所有** 内容类型，能搭配主流框架，并且 **可逆**。

|                                                                              | 覆盖范围                                          | 部署方式                            | 本地  | 可逆 |
|------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------|:-----:|:----:|
| **Headroom**                                                                 | 全部上下文 —— 工具、RAG、日志、文件、对话历史      | 代理 · 库 · 中间件 · MCP            | 是    | 是    |
| [RTK](https://github.com/rtk-ai/rtk)                                        | CLI 命令输出                                      | CLI 包装                            | 是    | 否    |
| [lean-ctx](https://github.com/yvgude/lean-ctx)                               | CLI 命令、MCP 工具、编辑器规则                     | CLI 包装 · MCP                      | 是    | 否    |
| [Compresr](https://compresr.ai)、[Token Co.](https://thetokencompany.ai)    | 发送给其官方 API 的文本                            | 远程 API 调用                       | 否    | 否    |
| OpenAI Compaction                                                            | 对话历史                                          | 服务商原生                          | 否    | 否    |

> **致谢。** Headroom 内置了优秀的 [RTK](https://github.com/rtk-ai/rtk) 工具来重写 shell 输出 —— `git show --short`、范围化的 `ls`、精简后的安装日志。非常感谢 RTK 团队，它已经成为我们工具链中不可或缺的一环，而 Headroom 会进一步压缩 RTK 下游的一切。你也可以让 Headroom 切换到 [lean-ctx](https://github.com/yvgude/lean-ctx) 作为 CLI 上下文工具：在执行 `headroom wrap ...` 前设置 `HEADROOM_CONTEXT_TOOL=lean-ctx` 即可。

## 参与贡献

```bash
git clone https://github.com/chopratejas/headroom.git && cd headroom
pip install -e ".[dev]" && pytest
```

`.devcontainer/` 里提供了默认配置与 `memory-stack`（含 Qdrant 与 Neo4j）。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 社区

- **[实时榜单](https://headroomlabs.ai/dashboard)** —— 已累计节省 600 亿+ tokens，仍在增长。
- **[Discord](https://discord.gg/yRmaUNpsPJ)** —— 提问、反馈、踩坑分享。
- **[HuggingFace 上的 Kompress-base](https://huggingface.co/chopratejas/kompress-base)** —— 驱动文本压缩的模型。

## 许可证

Apache 2.0 —— 详见 [LICENSE](LICENSE)。
